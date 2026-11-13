#ifndef ST7735_H
#define ST7735_H

#include "main.h"
#include <stdint.h>

/* VERIFY: Match these to your CubeMX pin names */
#define ST7735_CS_PORT   GPIOA
#define ST7735_CS_PIN    GPIO_PIN_11
#define ST7735_DC_PORT   GPIOA
#define ST7735_DC_PIN    GPIO_PIN_7
#define ST7735_RST_PORT  GPIOA
#define ST7735_RST_PIN   GPIO_PIN_12

#define ST7735_WIDTH  128
#define ST7735_HEIGHT 160

void ST7735_Init(void);
void ST7735_SetWindow(uint8_t x0, uint8_t y0, uint8_t x1, uint8_t y1);
void ST7735_DrawColorSolid(uint16_t color);
void ST7735_DrawBufferDMA(uint8_t x, uint8_t y, uint8_t w, uint8_t h, uint8_t *buffer);
uint8_t ST7735_IsBusy(void);

#endif
