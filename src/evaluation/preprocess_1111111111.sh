#!/bin/bash

# 参数检查
if [ $# -lt 3 ]; then
    echo "Usage: preprocess.sh <hypothesis-CTM-file> <tmp-cmt-file> <output-file>"
    exit 1
fi

input_file=$1
tmp_file=$2
output_file=$3

# 减少不必要的输出
echo "Starting preprocessing..."

# 简化处理流程
cat "${input_file}" | sed -E 's,loc-|cl-|qu-|poss-|lh-||__EMOTION__|__PU__|__LEFTHAND__,,g; s,S0NNE,SONNE,g; s,HABEN2,HABEN,g; s,WIE AUSSEHEN,WIE-AUSSEHEN,g; s,ZEIGEN ,ZEIGEN-BILDSCHIRM ,g; s,ZEIGEN$,ZEIGEN-BILDSCHIRM,g; s,^\([A-Z]\) \([A-Z][+ ]\),\1+\2,g; s,[ +]\([A-Z]\) \([A-Z]\) , \1+\2 ,g; s,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g; s,\([ +]SCH\) \([A-Z][ +]\),\1+\2,g; s,\([ +]NN\) \([A-Z][ +]\),\1+\2,g; s,\([ +][A-Z]\) \(NN[ +]\),\1+\2,g; s,\([ +][A-Z]\) \([A-Z]\)$,\1+\2,g; s,\([A-Z][A-Z]\)RAUM,\1,g; s,-PLUSPLUS,,g' | perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;' | grep -v "__LEFTHAND__" | grep -v "__EPENTHESIS__" | grep -v "__EMOTION__" > "${tmp_file}"

# 处理空结果
cat "${tmp_file}" | sed -e 's,\s*$,,' | awk 'BEGIN{lastID="";lastRow=""}{if (lastID!=$1 && cnt[lastID]<1 && lastRow!=""){print lastRow" [EMPTY]";}if ($5!=""){cnt[$1]+=1;print $0;}lastID=$1;lastRow=$0}' | sort -k1,1 -k3,3 > "${output_file}"

# 清理临时文件
rm "${tmp_file}"

# 输出完成信息
echo "$(date '+%Y-%m-%d %H:%M:%S') Preprocessing finished."
