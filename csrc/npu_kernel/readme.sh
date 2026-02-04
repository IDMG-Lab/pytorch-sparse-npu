#!/bin/bash
set -e

echo "======================"
echo " Step 1: run build.sh "
echo "======================"

if [ ! -f "./build.sh" ]; then
    echo "[ERROR] build.sh not found in current directory!"
    exit 1
fi

bash build.sh


echo ""
echo "============================================="
echo " Step 2: find and run build_out/custom*.run "
echo "============================================="

RUN_DIR="./build_out"
if [ ! -d "$RUN_DIR" ]; then
    echo "[ERROR] build_out directory not found!"
    exit 1
fi

# 找到第一个匹配 custom*.run 的文件
CUSTOM_RUN=$(find "$RUN_DIR" -maxdepth 1 -type f -name "custom*.run" | head -n 1)

if [ -z "$CUSTOM_RUN" ]; then
    echo "[ERROR] No custom*.run file found in build_out!"
    exit 1
fi

echo "Found run file: $CUSTOM_RUN"
chmod +x "$CUSTOM_RUN"
"$CUSTOM_RUN"

#这里只需要部署到acl算子就可以了，不需要在这里调用中间层的东西

# echo ""
# echo "======================="
# echo " Step 3: run run.sh "
# echo "======================="

# if [ ! -f "./run.sh" ]; then
#     echo "[ERROR] run.sh not found in current directory!"
#     exit 1
# fi

# chmod +x run.sh
# bash run.sh 1


# echo ""
# echo "========== All Done =========="
