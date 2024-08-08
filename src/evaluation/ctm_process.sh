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

# 简化处理流程
if ! cat "${input_file}" | sed -E 's,loc-|cl-|qu-|poss-|lh-||__EMOTION__|__PU__|__LEFTHAND__,,g;
        s,S0NNE,SONNE,g;
        s,HABEN2,HABEN,g;
        s,WIE AUSSEHEN,WIE-AUSSEHEN,g;
        s,ZEIGEN ,ZEIGEN-BILDSCHIRM ,g;
        s,ZEIGEN$,ZEIGEN-BILDSCHIRM,g;
        s,^\([A-Z]\) \([A-Z][+ ]\),\1+\2,g;
        s,[ +]\([A-Z]\) \([A-Z]\) , \1+\2 ,g;
        s,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g;
        s,\([ +]SCH\) \([A-Z][ +]\),\1+\2,g;
        s,\([ +]NN\) \([A-Z][ +]\),\1+\2,g;
        s,\([ +][A-Z]\) \(NN[ +]\),\1+\2,g;
        s,\([ +][A-Z]\) \([A-Z]\)$,\1+\2,g;
        s,\([A-Z][A-Z]\)RAUM,\1,g;
        s,-PLUSPLUS,,g;
        # 添加 Perl 的替换逻辑
        s,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;
        # 去除特定标记
        /__LEFTHAND__/d;
        /__EPENTHESIS__/d;
        /__EMOTION__/d;
        s,\s*$,,' |
awk 'BEGIN{lastID="";lastRow=""}{if (lastID!=$1 && cnt[lastID]<1 && lastRow!=""){print lastRow" [EMPTY]";}if ($5!=""){cnt[$1]+=1;print $0;}lastID=$1;lastRow=$0}' |
sort -k1,1 -k3,3 > "${output_file}"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') CTM processing finished. Output to ${output_file}"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') Error during CTM processing."
    exit 1
fi
