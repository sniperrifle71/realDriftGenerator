from datasets import multiflowDataset, elecDataset, weatherDataset
from model import officialGRU
from util import OnlineClassificationEvaluation
import torch
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    agrawal_dset = multiflowDataset(csv_dir="./eval_dataset/AGRAWAL_p700_w100_l1000.csv")
    gru = officialGRU(input_dim=9, output_dim=2, memory_size=128)
    state_dict = torch.load('./model_parameter/agrawal_pretrained_GRU.pth', map_location='cpu')
    gru.load_state_dict(state_dict)
    agrawal_loss_record, agrawal_acc_record = OnlineClassificationEvaluation(gru, agrawal_dset)

    sine_dset = multiflowDataset(csv_dir="./eval_dataset/SINE_p700_w100_l1000.csv")
    gru = officialGRU(input_dim=2, output_dim=2, memory_size=128)
    state_dict = torch.load('./model_parameter/SINE_pretrained_GRU.pth', map_location='cpu')
    gru.load_state_dict(state_dict)
    sine_loss_record, sine_acc_record = OnlineClassificationEvaluation(gru, sine_dset)

    waveform_dset = multiflowDataset(csv_dir="./eval_dataset/WAVEFORM_p700_w100_l1000.csv")
    gru = officialGRU(input_dim=21, output_dim=3, memory_size=128)
    state_dict = torch.load('./model_parameter/waveform_pretrained_GRU.pth', map_location='cpu')
    gru.load_state_dict(state_dict)
    waveform_loss_record, waveform_acc_record = OnlineClassificationEvaluation(gru, waveform_dset)

    rbf_dset = multiflowDataset(csv_dir="./eval_dataset/RBF_p700_w100_l1000.csv")
    gru = officialGRU(input_dim=10, output_dim=2, memory_size=128)
    state_dict = torch.load('./model_parameter/rbf_pretrained_GRU.pth', map_location='cpu')
    gru.load_state_dict(state_dict)
    rbf_loss_record, rbf_acc_record = OnlineClassificationEvaluation(gru, rbf_dset)

    drift_dict = {700:(100,"middle")}
    real_dset = elecDataset(online=True,stream_length=1000, drift_dict=drift_dict)
    gru = officialGRU(input_dim=5, output_dim=2, memory_size=128)
    state_dict = torch.load('./model_parameter/elec_pretrained_GRU.pth', map_location='cpu')
    gru.load_state_dict(state_dict)
    elec_loss_record, elec_acc_record = OnlineClassificationEvaluation(gru, real_dset)

    drift_dict = {700:(100,"middle")}
    real_dset = weatherDataset(online=True,stream_length=1000, drift_dict=drift_dict)
    gru = officialGRU(input_dim=4, output_dim=5, memory_size=128)
    state_dict = torch.load('./model_parameter/weather_pretrained_GRU.pth', map_location='cpu')
    gru.load_state_dict(state_dict)
    weather_loss_record, weather_acc_record = OnlineClassificationEvaluation(gru, real_dset)

    elec_err_record = [1-acc for acc in elec_acc_record]
    weather_err_record = [1-acc for acc in weather_acc_record]
    agrawal_err_record = [1-acc for acc in agrawal_acc_record]
    sine_err_record = [1-acc for acc in sine_acc_record]
    rbf_err_record = [1-acc for acc in rbf_acc_record]
    waveform_err_record = [1-acc for acc in waveform_acc_record]


    sns.set_theme()
    colors = sns.color_palette("husl", 8)
    fig, ax = plt.subplots(figsize=(6, 4))

    # 绘制线条，并为每条线指定样式和颜色
    ax.plot(elec_err_record, label="real: elec", linestyle='-', color=colors[4], linewidth=2)  # 深蓝
    ax.plot(weather_err_record, label="real: weather", linestyle='-', color=colors[5], linewidth=2)  # 中蓝
    ax.plot(agrawal_err_record, label="agrawal", linestyle='--', color=colors[0], linewidth=2)  # 浅蓝
    ax.plot(sine_err_record, label="sine", linestyle='-.', color=colors[1], linewidth=2)  # 绿色
    ax.plot(rbf_err_record, label="rbf", linestyle=':', color=colors[2], linewidth=2)  # 紫色
    ax.plot(waveform_err_record, label="waveform", linestyle='--', color=colors[3], linewidth=2)  # 橙色

    # 添加垂直线，并设置其样式
    ax.axvline(x=700, linestyle='--', color='gray', linewidth=1)

    # 添加标题和轴标签
    ax.set_title('Error Rate Comparison of Different Generators')
    ax.set_xlabel('Time')
    ax.set_ylabel('Error Rate')

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)


    num1 = 1.05
    num2 = 0
    num3 = 3
    num4 = 0
    plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)

    # 设置坐标轴限制（如果需要）
    ax.set_ylim(0, 0.5)  # 例如，如果准确率在0到1之间

    # 突出显示 x=600 到 x=800 的位置
    ax.axvspan(650, 750, color='grey', alpha=0.5)

    # 显示图表
    plt.show()
