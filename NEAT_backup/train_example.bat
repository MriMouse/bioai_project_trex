@echo off
echo Starting NEAT T-Rex Training with Enhanced Saving...
echo.

REM 基础训练命令 - 训练100代，前30代使用固定种子，适应度达到2500时切换随机种子
echo Command 1: Basic training with seed strategy
python train_neat.py train --config neat-config.txt --generations 100 --fixed-seed-generations 30 --min-fitness-for-random 2500

echo.
echo Training commands completed!
echo.
echo Available files after training:
echo - saved_models/neat-best-checkpoint-genX-fitnessXXXX-YYYYMMDD_HHMMSS (complete checkpoint)
echo - saved_models/neat-best-genome-genX-fitnessXXXX-YYYYMMDD_HHMMSS.pkl (best genome)
echo - saved_models/neat-stats-genX-fitnessXXXX-YYYYMMDD_HHMMSS.pkl (training statistics)
echo - saved_models/neat-best-genome-overall.pkl (latest best genome for compatibility)
echo.
pause
