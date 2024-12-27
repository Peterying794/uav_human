# uav_human
基于DEGCN和MMCL-Action(FR-Head)的人体行为识别  
(FR-Head中一个辅助特征细化头，可以加在gcn网络上）
# 环境安装
pip install -e torchlight  
pip install torch_topological  
注意：mix_GCN.yml已经基本满足所有model运行的环境
# 数据预处理
MMCL-Action:  
mirtest.py用于合成测试集test.npz(x_test,y_test)和验证集val.npz(x_test,y_test)，mir.py用于合成训练集train.npz(x_train,y_train,x_test,y_test)  
DEGCN:  
用的比赛给的数据集（没进行预处理），加了个int64.py生成的0label  
# 训练  
MMCL-Action:  
python main.py --config config/uav/jmmcl.yaml --work-dir results/uav/jmmcl --cl-mode ST-Multi-Level --w-multi-cl-loss 0.1 0.2 0.5 1 --device 0  

python main.py --config config/uav/bmmcl.yaml --work-dir results/uav/bmmcl --cl-mode ST-Multi-Level --w-multi-cl-loss 0.1 0.2 0.5 1 --device 0  

python main.py --config config/uav/jmmmcl.yaml --work-dir results/uav/jmmmcl --cl-mode ST-Multi-Level --w-multi-cl-loss 0.1 0.2 0.5 1 --device 0  

python main.py --config config/uav/bmmmcl.yaml --work-dir results/uav/bmmmcl --cl-mode ST-Multi-Level --w-multi-cl-loss 0.1 0.2 0.5 1 --device 0      

