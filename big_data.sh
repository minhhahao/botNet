#! /bin/bash

# output tmp file
outputfile=/home/aar0npham/Documents/Coding/tob/database/temp/RC-
# The amount of RAM in byte
max=16000000000

# Check whether file exists
matches_exist () {
  [ $# -gt 1 ] || [ -e "$1" ]
}

if matches_exist /home/aar0npham/Documents/Coding/tob/database/temp/*.json; then
  echo File already exist, exiting ...
else
  # Split the file
  for filename in /home/aar0npham/Documents/Coding/tob/database/RC/*.json; do
    filesize=$(wc -c <"$filename")
    echo $filename   is currently being processed
    if [ $filesize -ge $max ]; then
        echo File is bigger than $max byte of RAM, start spliting
        split --bytes 3GB --numeric-suffixes --suffix-length=3 --additional-suffix=.json $filename $outputfile
    else
        echo File is large enough to handle with RAM
    fi
  done
fi
