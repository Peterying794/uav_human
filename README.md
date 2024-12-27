# **基于DEGCN和MMCL-Action(FR-Head)的人体行为识别**     
(FR-Head中一个辅助特征细化头，可以加在gcn网络上）  
# 一、环境安装  
pip install -e torchlight  
pip install torch_topological  
注意：mix_GCN.yml已经基本满足所有model运行的环境
# 二、数据预处理
MMCL-Action:  
mirtest.py用于合成测试集test.npz(x_test,y_test)和验证集val.npz(x_test,y_test)，mir.py用于合成训练集train.npz(x_train,y_train,x_test,y_test)  
DEGCN:  
用的比赛给的数据集（没进行预处理），加了个int64.py生成的0label  
# 三、训练  
## DEGCN:  
### degcn_model:  
python main.py --config config/nturgbd-cross-subject/degcnj.yaml --device 0  

python main.py --config config/nturgbd-cross-subject/degcnjm.yaml --device 0  

python main.py --config config/nturgbd-cross-subject/degcnb.yaml --device 0  

python main.py --config config/nturgbd-cross-subject/degcnbm.yaml --device 0  
  
### jbf_model:  
python main.py --config config/nturgbd-cross-subject/jbfj.yaml --device 0  

python main.py --config config/nturgbd-cross-subject/jbfjm.yaml --device 0  

python main.py --config config/nturgbd-cross-subject/jbfb.yaml --device 0  

python main.py --config config/nturgbd-cross-subject/jbfbm.yaml --device 0  
## MMCL-Action(FR-Head):  
python main.py --config config/uav/jmmcl.yaml --work-dir results/uav/jmmcl --cl-mode ST-Multi-Level --w-multi-cl-loss 0.1 0.2 0.5 1 --device 0  

python main.py --config config/uav/bmmcl.yaml --work-dir results/uav/bmmcl --cl-mode ST-Multi-Level --w-multi-cl-loss 0.1 0.2 0.5 1 --device 0  

python main.py --config config/uav/jmmmcl.yaml --work-dir results/uav/jmmmcl --cl-mode ST-Multi-Level --w-multi-cl-loss 0.1 0.2 0.5 1 --device 0  

python main.py --config config/uav/bmmmcl.yaml --work-dir results/uav/bmmmcl --cl-mode ST-Multi-Level --w-multi-cl-loss 0.1 0.2 0.5 1 --device 0  
# 四、推理  
## FR-Head+MMCL:  

python main.py --config /home/featurize/work/block/FR-Head/config/uav/jmmcl.yaml --work-dir results/uav/jmmcl --phase test --save-score True --weights /home/featurize/work/block/FR-Head/results/uav/jmmcl/runs-64-16704.pt --device 0  
python main.py --config /home/featurize/work/block/FR-Head/config/uav/bmmcl.yaml --work-dir results/uav/bmmcl --phase test --save-score True --weights /home/featurize/work/block/FR-Head/results/uav/bmmcl/runs-57-14877.pt --device 0  
python main.py --config /home/featurize/work/block/FR-Head/config/uav/jmmmcl.yaml --work-dir results/uav/jmmmcl --phase test --save-score True --weights /home/featurize/work/block/FR-Head/results/uav/jmmmcl/runs-52-13572.pt --device 0  

python main.py --config /home/featurize/work/block/FR-Head/config/uav/bmmmcl.yaml --work-dir results/uav/bmmmcl --phase test --save-score True --weights /home/featurize/work/block/FR-Head/results/uav/bmmmcl/runs-58-15138.pt --device 0  
## DEGCN:  
### degcn_model:  

python main.py --config config/nturgbd-cross-subject/degcnj.yaml --phase test --save-score True --weights /mnt/workspace/block/DeGCN_pytorch/work_dir/degcn_j_3d/20241101115114/epoch_97_24832.pt --device 0  

python main.py --config config/nturgbd-cross-subject/degcnjm.yaml --phase test --save-score True --weights /mnt/workspace/block/DeGCN_pytorch/work_dir/degcn_jm_3d/20241101120014/epoch_100_25600.pt --device 0  

python main.py --config config/nturgbd-cross-subject/degcnb.yaml --phase test --save-score True --weights /mnt/workspace/block/DeGCN_pytorch/work_dir/degcn_b_3d/20241101115114/epoch_97_24832.pt --device 0  

python main.py --config config/nturgbd-cross-subject/degcnbm.yaml --phase test --save-score True --weights /mnt/workspace/block/DeGCN_pytorch/work_dir/degcn_bm_3d/20241101120014/epoch_100_25600.pt --device 0    
### jbf_model:
python main.py --config config/nturgbd-cross-subject/jbfj.yaml --phase test --save-score True --weights /mnt/workspace/block/DeGCN_pytorch/work_dir/jbf_j_3d/20241101144303/epoch_95_24320.pt --device 0  

python main.py --config config/nturgbd-cross-subject/jbfjm.yaml --phase test --save-score True --weights /mnt/workspace/block/DeGCN_pytorch/work_dir/jbf_jm_3d/20241101144450/epoch_97_24832.pt --device 0  

python main.py --config config/nturgbd-cross-subject/jbfb.yaml --phase test --save-score True --weights /mnt/workspace/block/DeGCN_pytorch/work_dir/jbf_b_3d/20241101144303/epoch_95_24320.pt --device 0  

python main.py --config config/nturgbd-cross-subject/jbfbm.yaml --phase test --save-score True --weights /mnt/workspace/block/DeGCN_pytorch/work_dir/jbf_bm_3d/20241101144450/epoch_97_24832.pt --device 0  
注意更改权重的地址  
# 五、集成模型  
运行Ensemble.py，生成集成的pred.npy,其中rate依据val的pkl与label进行贝叶斯优化得出（weight.py)和参考原来项目中ensemble.py给的权重包括每次提交的成绩进行调整  
# 六、weight网盘地址见复现总说明.txt




