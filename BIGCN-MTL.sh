#!/bin/bash


emb_size=(64)
lr=('3e-3')
gpu='cuda:6'
reg_weight=('1e-2')
dataset=('JD_2')
stage=('1')
inner_weight=('0.0')

his_weight=('0.0001')
tao=('0.01')
kd_weight=('0.01')

# shellcheck disable=SC2068
for name in ${dataset[@]}
do
    for l in ${lr[@]}
    do
        for reg in ${reg_weight[@]}
        do
            for emb in ${emb_size[@]}
            do
              for his in ${his_weight[@]}
              do
                for kd in ${kd_weight[@]}
                do
                  for t in ${tao[@]}
                  do
                  echo 'start train: '$name 'lr: '$l 'reg: '$reg 'emb: '$emb gpu: $gpu
                  `
                      nohup python main.py \
                          --lr ${l} \
                          --reg_weight ${reg} \
                          --his_weight ${his} \
                          --kd_weight ${kd} \
                          --tao  ${t}\
                          --data_name $name \
                          --device $gpu \
                          --stage $stage\
                          --embedding_size $emb > "./nohup_log/${name}/_lr_${l}_reg_${reg}_emb_${emb}.log" 2>&1 &
                  `
                  echo 'train end: '$name 'lr: '$l 'reg: '$reg 'emb: '$emb gpu: $gpu
                  done
                done
              done
            done
        done
    done
done
