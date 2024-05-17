#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

// #include "conv.h"
#include "resnet.h"

/**
 * @brief ʹ��resnet��������⺯��
 *
 * @param[in] inp_data: �����ԭʼ��Ƶ����
 * @param[out] is_voice: �Ƿ��������
 * @return ִ��״̬������ ALGO_NORMAL ��ʾִ�гɹ�������ֵ��ʾִ��ʧ��
 */
int resnet_vad(Conv2dData* inp_data, bool* is_voice);