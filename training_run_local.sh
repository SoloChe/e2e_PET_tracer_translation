#!/bin/bash

for resample in 1
do
    for generator_width in 300 
    do
        for num_residual_blocks_generator in 6 7 
        do
            for discriminator_width in 300 
            do
                for num_residual_blocks_discriminator in 6 7 
                do
                    for lambda_cyc in 10 
                    do
                        for lambda_id in 5
                        do
                            for lambda_mc in 0
                            do
                                for seed in 30
                                do  
                                    
                                    # batch size and pool size for buffer
                                    batch_size=20
                                    pool_size=500

                                    # adding cl or dm
                                    add_CL=False
                                    add_DM=False

                                    # patch discriminator
                                    patch_size=85 #85: no cl and dm 86: with cl 88: with cl and dm
                                    num_patch=1


                                     python ./train_local/train.py --resample $resample --generator_width $generator_width --num_residual_blocks_generator $num_residual_blocks_generator --discriminator_width $discriminator_width --num_residual_blocks_discriminator $num_residual_blocks_discriminator --lambda_cyc $lambda_cyc --lambda_id $lambda_id --lambda_mc $lambda_mc --log_path ./logs_${add_CL}${add_DM}${resample}_batch_${batch_size}_pool_${pool_size}_patch_${patch_size}_${num_patch}_${seed} --lr 0.0002 --decay_epoch 100 --n_epochs 700  --sample_interval 10 --batch_size $batch_size --pool_size $pool_size --patch_size $patch_size --num_patch $num_patch --seed $seed --add_CL $add_CL --add_DM $add_DM
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done