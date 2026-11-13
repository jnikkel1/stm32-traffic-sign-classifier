#include "st7735.h"

extern SPI_HandleTypeDef hspi1;
volatile uint8_t spi_dma_busy = 0;

static void ST7735_WriteCmd(uint8_t cmd) {
    HAL_GPIO_WritePin(ST7735_DC_PORT, ST7735_DC_PIN, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(ST7735_CS_PORT, ST7735_CS_PIN, GPIO_PIN_RESET);
    HAL_SPI_Transmit(&hspi1, &cmd, 1, HAL_MAX_DELAY);
    HAL_GPIO_WritePin(ST7735_CS_PORT, ST7735_CS_PIN, GPIO_PIN_SET);
}

static void ST7735_WriteData(uint8_t *data, uint16_t len) {
    HAL_GPIO_WritePin(ST7735_DC_PORT, ST7735_DC_PIN, GPIO_PIN_SET);
    HAL_GPIO_WritePin(ST7735_CS_PORT, ST7735_CS_PIN, GPIO_PIN_RESET);
    HAL_SPI_Transmit(&hspi1, data, len, HAL_MAX_DELAY);
    HAL_GPIO_WritePin(ST7735_CS_PORT, ST7735_CS_PIN, GPIO_PIN_SET);
}

void ST7735_Init(void) {
    // Hardware Reset
    HAL_GPIO_WritePin(ST7735_RST_PORT, ST7735_RST_PIN, GPIO_PIN_RESET);
    HAL_Delay(50);
    HAL_GPIO_WritePin(ST7735_RST_PORT, ST7735_RST_PIN, GPIO_PIN_SET);
    HAL_Delay(50);

    /* VERIFY IN DATASHEET: This is a standard ST7735S init sequence.
       Depending on your specific screen tab color (red/green/black), MADCTL and offsets might need tweaking */
    ST7735_WriteCmd(0x11); // Sleep Out
    HAL_Delay(120);

    ST7735_WriteCmd(0x3A); // Color Mode
    uint8_t color_mode = 0x05; // 16-bit RGB565
    ST7735_WriteData(&color_mode, 1);

    ST7735_WriteCmd(0x36); // Memory Access Control (MADCTL)
    uint8_t madctl = 0x00;
    ST7735_WriteData(&madctl, 1);

    ST7735_WriteCmd(0x29); // Display ON
}

void ST7735_SetWindow(uint8_t x0, uint8_t y0, uint8_t x1, uint8_t y1) {
    ST7735_WriteCmd(0x2A); // Column Address Set (CASET)
    uint8_t data_x[] = {0x00, x0, 0x00, x1};
    ST7735_WriteData(data_x, 4);

    ST7735_WriteCmd(0x2B); // Row Address Set (RASET)
    uint8_t data_y[] = {0x00, y0, 0x00, y1};
    ST7735_WriteData(data_y, 4);

    ST7735_WriteCmd(0x2C); // Memory Write (RAMWR)
}

void ST7735_DrawBufferDMA(uint8_t x, uint8_t y, uint8_t w, uint8_t h, uint8_t *buffer) {
    while(spi_dma_busy); // Wait for previous transfer
    ST7735_SetWindow(x, y, x + w - 1, y + h - 1);

    spi_dma_busy = 1;
    HAL_GPIO_WritePin(ST7735_DC_PORT, ST7735_DC_PIN, GPIO_PIN_SET);
    HAL_GPIO_WritePin(ST7735_CS_PORT, ST7735_CS_PIN, GPIO_PIN_RESET);

    // Transfer total bytes (w * h * 2 bytes per pixel)
    HAL_SPI_Transmit_DMA(&hspi1, buffer, w * h * 2);
}

uint8_t ST7735_IsBusy(void) {
    return spi_dma_busy;
}

// Override the weak HAL callback to release CS and clear busy flag
void HAL_SPI_TxCpltCallback(SPI_HandleTypeDef *hspi) {
    if (hspi->Instance == SPI1) {
        HAL_GPIO_WritePin(ST7735_CS_PORT, ST7735_CS_PIN, GPIO_PIN_SET);
        spi_dma_busy = 0;
    }
}
