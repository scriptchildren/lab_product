#!/bin/bash

#This program will format the data into words only
#i hope to read the file here

text_shaped(){

echo "plz tell me name of the file"
read file_str
echo "what name would you like to change?"
read file_changed
echo "OK! Print the length of the characters."
cat ${file_str} | sed -e 's/ /\n/g' -e 's/[\.,]//g' > ${file_changed}

}

text_shaped

#This program inputs the formatted data
#into a while statement to display
#the number of characters and their words.

#i hope to caluculate parcentage 

calc_len_amount(){

while read line
do
  echo ${#line} ${line}
  declare -i amount+=${#line}
done < ${file_changed}

echo ${amount}
}

calc_len_amount

echo "finish formatting and outputting the number of characters"
#In this program, i want to sort the words in order
# of their length and calculate the average length of the words.

echo "caluculate redability!"
num_l=$(calc_len_amount | sort -n | nl | awk 'END{print $1}')
#words_num=$(cat num_words.txt | awk 'NR==120' | awk '{print $1}')
words_amount=$(calc_len_amount | awk 'END{print $1}')
#echo $num_l $words_amount

result=$(awk "BEGIN {print $((${num_l}*100/${words_amount})) }")

echo "readability is ${result}%"

#done < $file_changed
