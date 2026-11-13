#ifndef SCCB_H
#define SCCB_H

#include "main.h"

#define SCCB_SDA_PORT GPIOC
#define SCCB_SDA_PIN  GPIO_PIN_10
#define SCCB_SCL_PORT GPIOC
#define SCCB_SCL_PIN  GPIO_PIN_12

#define OV7670_ADDR_WRITE 0x42
#define OV7670_ADDR_READ  0x43

void SCCB_Init(void);
uint8_t SCCB_ReadReg(uint8_t reg, uint8_t *val);
uint8_t SCCB_WriteReg(uint8_t reg, uint8_t data);

#endif
