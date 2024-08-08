#!/bin/bash

# 检查参数个数
if [ $# -lt 2 ]; then
    echo "Usage: $0 <hypothesis-CTM-file> <output-file>"
    echo "Where:"
    echo "  <hypothesis-CTM-file>: The input CTM file containing hypotheses."
    echo "  <output-file>: The output file where the processed data will be written."
    exit 1
fi

# 获取参数
input_file=$1
output_file=$2

# 检查输入文件是否存在且可读
if [ ! -f "$input_file" ] || [ ! -r "$input_file" ]; then
    echo "Error: Input file '$input_file' does not exist or is not readable."
    exit 1
fi

# 检查输出文件是否已存在
if [ -e "$output_file" ]; then
    echo "Warning: Output file '$output_file' already exists and will be overwritten."
fi

# 减少不必要的输出
echo "$(date '+%Y-%m-%d %H:%M:%S') Starting CTM processing..."

# 处理输入文件中的数据
cat "${input_file}" | \
    # 移除特定的错误识别和格式化文本
    sed -e 's,loc-||cl-||qu-||poss-||lh-||__EMOTION__||__PU__||__LEFTHAND__,,g' \
    -e 's,S0NNE,SONNE,g' -e 's,HABEN2,HABEN,g' -e 's,WIE AUSSEHEN,WIE-AUSSEHEN,g' \
    -e 's,ZEIGEN ,ZEIGEN-BILDSCHIRM ,g' -e 's,ZEIGEN$,ZEIGEN-BILDSCHIRM,g' \
    -e 's,^\([A-Z]\) \([A-Z][+ ]\),\1+\2,g' -e 's,[ +]\([A-Z]\) \([A-Z]\) , \1+\2 ,g' \
    -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g' -e 's,\([ +]SCH\) \([A-Z][ +]\),\1+\2,g' \
    -e 's,\([ +]NN\) \([A-Z][ +]\),\1+\2,g' -e 's,\([ +][A-Z]\) \(NN[ +]\),\1+\2,g' \
    -e 's,\([ +][A-Z]\) \([A-Z]\)$,\1+\2,g' -e 's,\([A-Z][A-Z]\)RAUM,\1,g' -e 's,-PLUSPLUS,,g' | \
    # 处理大小写和重复的单词
    perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g; print;' | \
    # 过滤掉特定的行
    grep -v "__LEFTHAND__" | grep -v "__EPENTHESIS__" | grep -v "__EMOTION__" | \
    # 清理行尾的空格
    sed -e 's,\s*$,,' | \
    # 处理空行和重复的记录
    awk 'BEGIN{lastID="";lastRow=""}{if (lastID!=$1 && cnt[lastID]<1 && lastRow!=""){print lastRow" [EMPTY]";}if ($5!=""){cnt[$1]+=1;print $0;}lastID=$1;lastRow=$0}' | \
    # 根据第一和第三个字段排序
    sort -k1,1 -k3,3 > "${output_file}"

# 检查命令的执行结果
if [ $? -eq 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') CTM processing finished. Output to ${output_file}"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') Error during CTM processing."
    exit 1
fi
