#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

// #include "conv.h"
#include "resnet.h"

/**
 * @brief 使用resnet的人声检测函数
 *
 * @param[in] inp_data: 输入的原始音频数据
 * @param[out] is_voice: 是否包含人声
 * @return 执行状态，返回 ALGO_NORMAL 表示执行成功，其他值表示执行失败
 */
int resnet_vad(Conv2dData* inp_data, bool* is_voice);