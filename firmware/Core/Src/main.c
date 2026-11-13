/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Real-Time Embedded Traffic Sign Recognition Pipeline
  *                   (STM32F446RE + OV7670 DCMI/DMA + X-CUBE-AI INT8 + ST7735)
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "app_x-cube-ai.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include "sccb.h"
#include "ov7670.h"
#include "st7735.h"
#include "network.h"
#include "network_data.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define CAM_W               128
#define CAM_H               120
#define CAM_FRAME_WORDS     ((CAM_W * CAM_H * 2) / 4) /* 7,680 32-bit words (30,720 bytes) */

#define ROI_X_OFFSET        16
#define ROI_Y_OFFSET        12
#define ROI_SIZE            96

#define MODEL_IMG_W         48
#define MODEL_IMG_H         48
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
CRC_HandleTypeDef hcrc;

DCMI_HandleTypeDef hdcmi;
DMA_HandleTypeDef hdma_dcmi;

SPI_HandleTypeDef hspi1;
DMA_HandleTypeDef hdma_spi1_tx;

UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
/* 128x120 RGB565 frame buffer (30,720 bytes) */
uint32_t frame_buffer[CAM_FRAME_WORDS];

/* X-CUBE-AI aligned activation scratchpad and I/O tensor buffers */
static ai_handle gtsrb_network = AI_HANDLE_NULL;
AI_ALIGNED(32) static ai_u8 ai_activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];
AI_ALIGNED(32) static ai_u8 ai_in_data[AI_NETWORK_IN_1_SIZE_BYTES];
AI_ALIGNED(32) static ai_u8 ai_out_data[AI_NETWORK_OUT_1_SIZE_BYTES];

static ai_buffer ai_input_buf[1];
static ai_buffer ai_output_buf[1];

/* Dedicated 128x20 RGB565 buffer for the top UI banner (5,120 bytes) */
static uint8_t banner_buf[128 * 20 * 2];
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_SPI1_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_DCMI_Init(void);
static void MX_CRC_Init(void);
/* USER CODE BEGIN PFP */
const char* GTSRB_GetLabel(int class_id);
void GTSRB_AI_Init(void);
void GTSRB_Preprocess_Frame(const uint16_t *src_rgb565);
int  GTSRB_Run_Inference(int *out_confidence);
void LCD_Draw_ROI_Box(uint8_t *fb_bytes);
void LCD_Draw_Sign_Banner(int class_id, int confidence);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* Compact 5x7 ASCII bitmap font (printable characters 32 ' ' to 90 'Z') */
static const uint8_t font5x7[][5] = {
    {0x00,0x00,0x00,0x00,0x00}, {0x00,0x00,0x5F,0x00,0x00}, {0x00,0x07,0x00,0x07,0x00},
    {0x14,0x7F,0x14,0x7F,0x14}, {0x24,0x2A,0x7F,0x2A,0x12}, {0x23,0x13,0x08,0x64,0x62},
    {0x36,0x49,0x55,0x22,0x50}, {0x00,0x05,0x03,0x00,0x00}, {0x00,0x1C,0x22,0x41,0x00},
    {0x00,0x41,0x22,0x1C,0x00}, {0x14,0x08,0x3E,0x08,0x14}, {0x08,0x08,0x3E,0x08,0x08},
    {0x00,0x50,0x30,0x00,0x00}, {0x08,0x08,0x08,0x08,0x08}, {0x00,0x60,0x60,0x00,0x00},
    {0x20,0x10,0x08,0x04,0x02}, {0x3E,0x51,0x49,0x45,0x3E}, {0x00,0x42,0x7F,0x40,0x00},
    {0x42,0x61,0x51,0x49,0x46}, {0x21,0x41,0x45,0x4B,0x31}, {0x18,0x14,0x12,0x7F,0x10},
    {0x27,0x45,0x45,0x45,0x39}, {0x3C,0x4A,0x49,0x49,0x30}, {0x01,0x71,0x09,0x05,0x03},
    {0x36,0x49,0x49,0x49,0x36}, {0x06,0x49,0x49,0x29,0x1E}, {0x00,0x36,0x36,0x00,0x00},
    {0x00,0x56,0x36,0x00,0x00}, {0x08,0x14,0x22,0x41,0x00}, {0x14,0x14,0x14,0x14,0x14},
    {0x00,0x41,0x22,0x14,0x08}, {0x02,0x01,0x51,0x09,0x06}, {0x32,0x49,0x79,0x41,0x3E},
    {0x7E,0x11,0x11,0x11,0x7E}, {0x7F,0x49,0x49,0x49,0x36}, {0x3E,0x41,0x41,0x41,0x22},
    {0x7F,0x41,0x41,0x22,0x1C}, {0x7F,0x49,0x49,0x49,0x41}, {0x7F,0x09,0x09,0x09,0x01},
    {0x3E,0x41,0x49,0x49,0x7A}, {0x7F,0x08,0x08,0x08,0x7F}, {0x00,0x41,0x7F,0x41,0x00},
    {0x20,0x40,0x41,0x3F,0x01}, {0x7F,0x08,0x14,0x22,0x41}, {0x7F,0x40,0x40,0x40,0x40},
    {0x7F,0x02,0x0C,0x02,0x7F}, {0x7F,0x04,0x08,0x10,0x7F}, {0x3E,0x41,0x41,0x41,0x3E},
    {0x7F,0x09,0x09,0x09,0x06}, {0x3E,0x41,0x51,0x21,0x5E}, {0x7F,0x09,0x19,0x29,0x46},
    {0x46,0x49,0x49,0x49,0x31}, {0x01,0x01,0x7F,0x01,0x01}, {0x3F,0x40,0x40,0x40,0x3F},
    {0x1F,0x20,0x40,0x20,0x1F}, {0x3F,0x40,0x38,0x40,0x3F}, {0x63,0x14,0x08,0x14,0x63},
    {0x07,0x08,0x70,0x08,0x07}, {0x61,0x51,0x49,0x45,0x43}
};

/**
 * @brief Maps GTSRB class indices (0..42) to human-readable traffic sign labels.
 */
const char* GTSRB_GetLabel(int class_id) {
    static const char* labels[43] = {
        "20 km/h", "30 km/h", "50 km/h", "60 km/h", "70 km/h",
        "80 km/h", "End 80", "100 km/h", "120 km/h", "No Passing",
        "No Pass >3.5t", "Right of Way", "Priority Road", "Yield", "STOP",
        "No Vehicles", "No >3.5t", "No Entry", "General Caution", "Curve Left",
        "Curve Right", "Double Curve", "Bumpy Road", "Slippery Road", "Road Narrows",
        "Road Work", "Traffic Signals", "Pedestrians", "Children", "Bicycles",
        "Beware Ice/Snow", "Wild Animals", "End Limits", "Turn Right", "Turn Left",
        "Ahead Only", "Straight/Right", "Straight/Left", "Keep Right", "Keep Left",
        "Roundabout", "End No Passing", "End No Pass >3.5t"
    };
    if (class_id >= 0 && class_id < 43) {
        return labels[class_id];
    }
    return "Unknown";
}

/**
 * @brief Initializes the quantized INT8 CNN in X-CUBE-AI and binds I/O tensor buffers.
 */
void GTSRB_AI_Init(void) {
    __HAL_RCC_CRC_CLK_ENABLE();

    const ai_handle act_addr[] = { ai_activations };
    ai_error err = ai_network_create_and_init(&gtsrb_network, act_addr, NULL);
    if (err.type != AI_ERROR_NONE) {
        printf("AI Init Error: type=%d, code=%d\r\n", err.type, err.code);
        return;
    }

    const ai_buffer *in_desc  = ai_network_inputs_get(gtsrb_network, NULL);
    const ai_buffer *out_desc = ai_network_outputs_get(gtsrb_network, NULL);
    ai_input_buf[0]  = in_desc[0];
    ai_output_buf[0] = out_desc[0];

    ai_input_buf[0].data  = AI_HANDLE_PTR(ai_in_data);
    ai_output_buf[0].data = AI_HANDLE_PTR(ai_out_data);
}

/**
 * @brief Downsamples the centered 96x96 RGB565 ROI to 48x48 RGB888 using a 2x2
 *        box filter and quantizes pixels to signed INT8 [-128, 127].
 */
void GTSRB_Preprocess_Frame(const uint16_t *src_rgb565) {
    int8_t *dst_i8 = (int8_t *)ai_in_data;
    uint32_t idx = 0;

    for (int y = 0; y < MODEL_IMG_H; y++) {
        int sy = ROI_Y_OFFSET + (y * 2);
        for (int x = 0; x < MODEL_IMG_W; x++) {
            int sx = ROI_X_OFFSET + (x * 2);

            uint32_t sr = 0, sg = 0, sb = 0;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    const uint8_t *p = (const uint8_t *)&src_rgb565[(sy + dy) * CAM_W + (sx + dx)];
                    uint16_t pix = ((uint16_t)p[0] << 8) | p[1];

                    sr += ((pix >> 11) & 0x1F) << 3;
                    sg += ((pix >> 5)  & 0x3F) << 2;
                    sb += ( pix        & 0x1F) << 3;
                }
            }

            int r = (int)(sr >> 2);
            int g = (int)(sg >> 2);
            int b = (int)(sb >> 2);

            dst_i8[idx++] = (int8_t)(r - 128);
            dst_i8[idx++] = (int8_t)(g - 128);
            dst_i8[idx++] = (int8_t)(b - 128);
        }
    }
}

/**
 * @brief Executes forward pass and returns predicted class index and confidence (0..100%).
 */
int GTSRB_Run_Inference(int *out_confidence) {
    ai_i32 batch = ai_network_run(gtsrb_network, &ai_input_buf[0], &ai_output_buf[0]);
    if (batch != 1) {
        ai_error err = ai_network_get_error(gtsrb_network);
        printf("AI Run Error: code=%d, type=%d\r\n", err.code, err.type);
        return -1;
    }

    const int8_t *out = (const int8_t *)ai_out_data;
    uint32_t best = 0;
    for (uint32_t i = 1; i < AI_NETWORK_OUT_1_SIZE; i++) {
        if (out[i] > out[best]) {
            best = i;
        }
    }

    const ai_buffer_meta_info *mi = ai_output_buf[0].meta_info;
    float scale = AI_BUFFER_META_INFO_INTQ_GET_SCALE(mi, 0);
    int zp      = AI_BUFFER_META_INFO_INTQ_GET_ZEROPOINT(mi, 0);

    float prob = ((int)out[best] - zp) * scale;
    int conf   = (int)(prob * 100.0f);
    if (conf < 0)   conf = 0;
    if (conf > 100) conf = 100;

    *out_confidence = conf;
    return (int)best;
}

/**
 * @brief Draws a 1-pixel green targeting box around the 96x96 ROI on the LCD frame.
 */
void LCD_Draw_ROI_Box(uint8_t *fb_bytes) {
    const int x0 = ROI_X_OFFSET;
    const int x1 = ROI_X_OFFSET + ROI_SIZE - 1;
    const int y0 = ROI_Y_OFFSET;
    const int y1 = ROI_Y_OFFSET + ROI_SIZE - 1;

    for (int x = x0; x <= x1; x++) {
        fb_bytes[(y0 * CAM_W + x) * 2]     = 0x07;
        fb_bytes[(y0 * CAM_W + x) * 2 + 1] = 0xE0;
        fb_bytes[(y1 * CAM_W + x) * 2]     = 0x07;
        fb_bytes[(y1 * CAM_W + x) * 2 + 1] = 0xE0;
    }
    for (int y = y0; y <= y1; y++) {
        fb_bytes[(y * CAM_W + x0) * 2]     = 0x07;
        fb_bytes[(y * CAM_W + x0) * 2 + 1] = 0xE0;
        fb_bytes[(y * CAM_W + x1) * 2]     = 0x07;
        fb_bytes[(y * CAM_W + x1) * 2 + 1] = 0xE0;
    }
}

/**
 * @brief Renders the classification label and confidence into the top 128x20 LCD bar.
 */
void LCD_Draw_Sign_Banner(int class_id, int confidence) {
    char text[24];
    uint16_t bg_color = 0x0000;
    uint16_t fg_color = 0xFFFF;

    if (class_id >= 0 && confidence >= 50) {
        snprintf(text, sizeof(text), "%s %d%%", GTSRB_GetLabel(class_id), confidence);
        bg_color = (confidence >= 75) ? 0x0320 : 0x4200;
        fg_color = 0xFFFF;
    } else {
        snprintf(text, sizeof(text), "SCANNING...");
        bg_color = 0x0000;
        fg_color = 0x07E0;
    }

    uint8_t bg_hi = (uint8_t)(bg_color >> 8), bg_lo = (uint8_t)(bg_color & 0xFF);
    uint8_t fg_hi = (uint8_t)(fg_color >> 8), fg_lo = (uint8_t)(fg_color & 0xFF);

    for (int i = 0; i < 128 * 20; i++) {
        banner_buf[i * 2]     = bg_hi;
        banner_buf[i * 2 + 1] = bg_lo;
    }

    int cursor_x = 4;
    const int start_y = 6;

    for (int i = 0; text[i] != '\0' && (cursor_x + 6) < 128; i++) {
        char c = text[i];
        if (c >= 'a' && c <= 'z') c -= 32;
        if (c < 32 || c > 90) c = ' ';

        const uint8_t *glyph = font5x7[c - 32];
        for (int col = 0; col < 5; col++) {
            uint8_t bits = glyph[col];
            for (int row = 0; row < 7; row++) {
                if (bits & (1 << row)) {
                    int px = (start_y + row) * 128 + (cursor_x + col);
                    banner_buf[px * 2]     = fg_hi;
                    banner_buf[px * 2 + 1] = fg_lo;
                }
            }
        }
        cursor_x += 6;
    }

    ST7735_DrawBufferDMA(0, 0, 128, 20, banner_buf);
    uint32_t t0 = HAL_GetTick();
    while (ST7735_IsBusy()) {
        if ((HAL_GetTick() - t0) > 50) break;
    }
}

#ifdef __GNUC__
  #define PUTCHAR_PROTOTYPE int __io_putchar(int ch)
#else
  #define PUTCHAR_PROTOTYPE int fputc(int ch, FILE *f)
#endif

PUTCHAR_PROTOTYPE {
    HAL_UART_Transmit(&huart2, (uint8_t *)&ch, 1, 0xFFFF);
    return ch;
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_SPI1_Init();
  MX_USART2_UART_Init();
  MX_DCMI_Init();
  MX_CRC_Init();

  /* USER CODE BEGIN 2 */
  /* Enforce DCMI Synchronization Polarities */
  hdcmi.Init.VSPolarity  = DCMI_VSPOLARITY_HIGH;
  hdcmi.Init.HSPolarity  = DCMI_HSPOLARITY_LOW;
  hdcmi.Init.PCKPolarity = DCMI_PCKPOLARITY_RISING;
  HAL_DCMI_Init(&hdcmi);

  /* Set DCMI DMA2 Stream1 to VERY_HIGH priority to prevent AHB overrun against SPI1 */
  HAL_DMA_DeInit(&hdma_dcmi);
  hdma_dcmi.Init.Mode                = DMA_NORMAL;
  hdma_dcmi.Init.Priority            = DMA_PRIORITY_VERY_HIGH;
  hdma_dcmi.Init.FIFOMode            = DMA_FIFOMODE_DISABLE;
  hdma_dcmi.Init.MemDataAlignment    = DMA_MDATAALIGN_WORD;
  hdma_dcmi.Init.PeriphDataAlignment = DMA_PDATAALIGN_WORD;
  HAL_DMA_Init(&hdma_dcmi);

  ST7735_Init();
  SCCB_Init();
  OV7670_Init();

  /* Hardware crop 160x120 QQVGA stream to 128x120 RGB565 (256 bytes/line) */
  HAL_DCMI_ConfigCrop(&hdcmi, 32, 0, 255, 119);
  HAL_DCMI_EnableCrop(&hdcmi);

  GTSRB_AI_Init();
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
      /* 1. Capture single 128x120 RGB565 frame via DCMI DMA */
      HAL_DCMI_Stop(&hdcmi);
      __HAL_DCMI_CLEAR_FLAG(&hdcmi, DCMI_FLAG_FRAMERI | DCMI_FLAG_OVRRI |
                                    DCMI_FLAG_ERRRI   | DCMI_FLAG_VSYNCRI |
                                    DCMI_FLAG_LINERI);

      HAL_DCMI_Start_DMA(&hdcmi, DCMI_MODE_SNAPSHOT, (uint32_t)frame_buffer, CAM_FRAME_WORDS);

      uint32_t tickstart = HAL_GetTick();
      while (hdma_dcmi.Instance->NDTR > 0) {
          if ((HAL_GetTick() - tickstart) > 500) {
              break;
          }
      }
      HAL_DCMI_Stop(&hdcmi);

      /* 2. Preprocess 96x96 ROI into 48x48x3 INT8 input tensor before drawing UI overlay */
      GTSRB_Preprocess_Frame((const uint16_t *)frame_buffer);

      /* 3. Draw ROI targeting overlay and stream video buffer to ST7735 via SPI1 DMA */
      LCD_Draw_ROI_Box((uint8_t *)frame_buffer);
      ST7735_DrawBufferDMA(0, 20, CAM_W, CAM_H, (uint8_t *)frame_buffer);

      uint32_t spi_tick = HAL_GetTick();
      while (ST7735_IsBusy()) {
          if ((HAL_GetTick() - spi_tick) > 200) {
              break;
          }
      }

      /* 4. Execute CNN inference and update top LCD banner + UART telemetry */
      int confidence = 0;
      uint32_t t0 = HAL_GetTick();
      int class_id = GTSRB_Run_Inference(&confidence);
      uint32_t dt = HAL_GetTick() - t0;

      printf("Sign: [%2d] %-16s | Conf: %3d%% | Time: %lu ms\r\n",
             class_id, GTSRB_GetLabel(class_id), confidence, dt);

      LCD_Draw_Sign_Banner(class_id, confidence);
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 180;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Activate the Over-Drive mode
  */
  if (HAL_PWREx_EnableOverDrive() != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
  HAL_RCC_MCOConfig(RCC_MCO1, RCC_MCO1SOURCE_HSI, RCC_MCODIV_1);
}

/**
  * @brief CRC Initialization Function
  * @param None
  * @retval None
  */
static void MX_CRC_Init(void)
{

  /* USER CODE BEGIN CRC_Init 0 */

  /* USER CODE END CRC_Init 0 */

  /* USER CODE BEGIN CRC_Init 1 */

  /* USER CODE END CRC_Init 1 */
  hcrc.Instance = CRC;
  if (HAL_CRC_Init(&hcrc) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN CRC_Init 2 */

  /* USER CODE END CRC_Init 2 */

}

/**
  * @brief DCMI Initialization Function
  * @param None
  * @retval None
  */
static void MX_DCMI_Init(void)
{

  /* USER CODE BEGIN DCMI_Init 0 */

  /* USER CODE END DCMI_Init 0 */

  /* USER CODE BEGIN DCMI_Init 1 */

  /* USER CODE END DCMI_Init 1 */
  hdcmi.Instance = DCMI;
  hdcmi.Init.SynchroMode = DCMI_SYNCHRO_HARDWARE;
  hdcmi.Init.PCKPolarity = DCMI_PCKPOLARITY_RISING;
  hdcmi.Init.VSPolarity = DCMI_VSPOLARITY_HIGH;
  hdcmi.Init.HSPolarity = DCMI_HSPOLARITY_LOW;
  hdcmi.Init.CaptureRate = DCMI_CR_ALL_FRAME;
  hdcmi.Init.ExtendedDataMode = DCMI_EXTEND_DATA_8B;
  hdcmi.Init.JPEGMode = DCMI_JPEG_DISABLE;
  hdcmi.Init.ByteSelectMode = DCMI_BSM_ALL;
  hdcmi.Init.ByteSelectStart = DCMI_OEBS_ODD;
  hdcmi.Init.LineSelectMode = DCMI_LSM_ALL;
  hdcmi.Init.LineSelectStart = DCMI_OELS_ODD;
  if (HAL_DCMI_Init(&hdcmi) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN DCMI_Init 2 */

  /* USER CODE END DCMI_Init 2 */

}

/**
  * @brief SPI1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI1_Init(void)
{

  /* USER CODE BEGIN SPI1_Init 0 */

  /* USER CODE END SPI1_Init 0 */

  /* USER CODE BEGIN SPI1_Init 1 */

  /* USER CODE END SPI1_Init 1 */
  /* SPI1 parameter configuration*/
  hspi1.Instance = SPI1;
  hspi1.Init.Mode = SPI_MODE_MASTER;
  hspi1.Init.Direction = SPI_DIRECTION_2LINES;
  hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;
  hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;
  hspi1.Init.NSS = SPI_NSS_SOFT;
  hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_4;
  hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi1.Init.CRCPolynomial = 10;
  if (HAL_SPI_Init(&hspi1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI1_Init 2 */

  /* USER CODE END SPI1_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA2_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA2_Stream1_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream1_IRQn);
  /* DMA2_Stream3_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream3_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream3_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, LCD_DC_Pin|LCD_CS_Pin|LCD_RESET_Pin, GPIO_PIN_SET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOC, CAM_SDA_Pin|CAM_SCL_Pin, GPIO_PIN_SET);

  /*Configure GPIO pins : LCD_DC_Pin LCD_CS_Pin LCD_RESET_Pin */
  GPIO_InitStruct.Pin = LCD_DC_Pin|LCD_CS_Pin|LCD_RESET_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pin : PA8 */
  GPIO_InitStruct.Pin = GPIO_PIN_8;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF0_MCO;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pin : CAM_SDA_Pin */
  GPIO_InitStruct.Pin = CAM_SDA_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_OD;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
  HAL_GPIO_Init(CAM_SDA_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : CAM_SCL_Pin */
  GPIO_InitStruct.Pin = CAM_SCL_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
  HAL_GPIO_Init(CAM_SCL_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
