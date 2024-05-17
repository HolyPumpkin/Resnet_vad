#include "resnet_vad.h"

static ResNet resnet;



/**
 * @brief ʹ��resnet��������⺯��
 *
 * @param[in] inp_data: �����ԭʼ��Ƶ����
 * @param[out] is_voice: �Ƿ��������
 * @return ִ��״̬������ ALGO_NORMAL ��ʾִ�гɹ�������ֵ��ʾִ��ʧ��
 */
int resnet_vad(Conv2dData* inp_data, bool* is_voice)
{
    int ret = ALGO_NORMAL;

    // ����resnet����ṹ�岢��ʼ��
    initialize_resnet(&resnet);

    // ����resnetǰ�򴫲�����
    double output[2] = { 0 };
    ret = resnet_forward(inp_data, &resnet, output);
    if (ret != ALGO_NORMAL) {
        return ret;
    }

    // �ж��������Ƿ�Ϊ����
    *is_voice = (output[1] > output[0]);

    return ret;
}