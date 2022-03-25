#!/bin/bash


prefix=${PWD}/vantage6_config_files/node/

folders=("2class_class_imb/" "2class_IID/" "2class_sample_imb/" "2node_PCA/" "3node_PCA/" "3node_raw/" "4class_class_imb/" "4class_IID/" "4class_sample_imb/" "A2_class_imb/" "A2_IID/" "A2_sample_imb/")

address=$(ip a |grep -A 4 docker0: | grep -o 'inet.*' | cut -c 6-15)

#echo "$address"

for ((j = 0; j<=12; j++))
do
    for ((i = 0; i<=9; i++))
    do 
        file="${prefix}${folders[${j}]}org${i}.yaml"
        if test -f "$file"; then
            sed -i s,/path/to/base/folder/,${PWD},g ${file}
            sed -i s,~/Documents/afstuderen/vantage6/tasks," ",g ${file}
            sed -i s,server_url:,"server_url: http://${address}",g ${file}
        fi
        #sed -i s,/home/swier/Documents/afstuderen/,/path/to/base/,g ${file}
        #sed -i s,/home/swier/.local/share/vantage6/node/,/path/to/base/,g ${file}
        #sed -i s,"server_url: http://${address}",server_url:,g ${file}
        #sed -i s,"server_url: ",server_url:,g ${file}
    done
done


sed -i s,/absolute/path/to/folder/,${PWD}/,g vantage6_config_files/server/vantage_server.yaml



