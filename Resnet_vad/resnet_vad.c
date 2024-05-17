#include "resnet_vad.h"

static ResNet resnet;



/**
 * @brief 使用resnet的人声检测函数
 *
 * @param[in] inp_data: 输入的原始音频数据
 * @param[out] is_voice: 是否包含人声
 * @return 执行状态，返回 ALGO_NORMAL 表示执行成功，其他值表示执行失败
 */
int resnet_vad(Conv2dData* inp_data, bool* is_voice)
{
    int ret = ALGO_NORMAL;

    // 创建resnet网络结构体并初始化
    initialize_resnet(&resnet);

    // 调用resnet前向传播函数
    double output[2] = { 0 };
    ret = resnet_forward(inp_data, &resnet, output);
    if (ret != ALGO_NORMAL) {
        return ret;
    }

    // 判断输出结果是否为语音
    *is_voice = (output[1] > output[0]);

    return ret;
}