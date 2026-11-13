/**
  ******************************************************************************
  * @file           : sccb.c
  * @brief          : Bit-Banged OmniVision Serial Camera Control Bus (SCCB)
  ******************************************************************************
  */
#include "sccb.h"

#define SCL_H()    HAL_GPIO_WritePin(SCCB_SCL_PORT, SCCB_SCL_PIN, GPIO_PIN_SET)
#define SCL_L()    HAL_GPIO_WritePin(SCCB_SCL_PORT, SCCB_SCL_PIN, GPIO_PIN_RESET)
#define SDA_H()    HAL_GPIO_WritePin(SCCB_SDA_PORT, SCCB_SDA_PIN, GPIO_PIN_SET)
#define SDA_L()    HAL_GPIO_WritePin(SCCB_SDA_PORT, SCCB_SDA_PIN, GPIO_PIN_RESET)
#define SDA_READ() HAL_GPIO_ReadPin(SCCB_SDA_PORT, SCCB_SDA_PIN)

static void sccb_delay(void) {
    for (volatile int i = 0; i < 5000; i++);
}

static void SCCB_Start(void) {
    SDA_H(); SCL_H(); sccb_delay();
    SDA_L(); sccb_delay();
    SCL_L(); sccb_delay();
}

static void SCCB_Stop(void) {
    SDA_L(); sccb_delay();
    SCL_H(); sccb_delay();
    SDA_H(); sccb_delay();
}

static uint8_t SCCB_WriteByte(uint8_t data) {
    for (int i = 0; i < 8; i++) {
        if (data & 0x80) SDA_H(); else SDA_L();
        data <<= 1;
        sccb_delay(); SCL_H(); sccb_delay(); SCL_L(); sccb_delay();
    }
    SDA_H();
    sccb_delay(); SCL_H(); sccb_delay();
    uint8_t ack = SDA_READ();
    SCL_L(); sccb_delay();
    return ack;
}

static uint8_t SCCB_ReadByte(void) {
    uint8_t data = 0;
    SDA_H();
    for (int i = 0; i < 8; i++) {
        sccb_delay(); SCL_H(); sccb_delay();
        data <<= 1;
        if (SDA_READ()) data |= 0x01;
        SCL_L(); sccb_delay();
    }
    SDA_H(); sccb_delay(); SCL_H(); sccb_delay(); SCL_L(); sccb_delay();
    return data;
}

void SCCB_Init(void) {
    SDA_H();
    SCL_H();
    HAL_Delay(10);
}

uint8_t SCCB_ReadReg(uint8_t reg, uint8_t *val) {
    SCCB_Start();
    SCCB_WriteByte(OV7670_ADDR_WRITE);
    SCCB_WriteByte(reg);
    SCCB_Stop();

    sccb_delay();

    SCCB_Start();
    SCCB_WriteByte(OV7670_ADDR_READ);
    *val = SCCB_ReadByte();
    SCCB_Stop();
    return 0;
}

uint8_t SCCB_WriteReg(uint8_t reg, uint8_t data) {
    SCCB_Start();
    SCCB_WriteByte(OV7670_ADDR_WRITE);
    SCCB_WriteByte(reg);
    SCCB_WriteByte(data);
    SCCB_Stop();
    return 0;
}
