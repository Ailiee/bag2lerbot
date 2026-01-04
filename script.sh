#!/bin/bash

TARGET_DIR="${1:-.}"

echo "Opening permissions..."
chmod -R 777 "$TARGET_DIR"

total=$(find "$TARGET_DIR" -mindepth 2 -maxdepth 2 -type f -name "*.tar" | wc -l)
echo "Found $total .tar files"

counter_file="/tmp/extract_counter_$$"
echo "0" > "$counter_file"

find "$TARGET_DIR" -mindepth 2 -maxdepth 2 -type f -name "*.tar" | \
xargs -P $(nproc) -I {} bash -c '
  tarfile="{}"
  dir=$(dirname "$tarfile")
  counter_file="'"$counter_file"'"
  total='"$total"'
  
  first_entry=$(tar -tf "$tarfile" 2>/dev/null | head -1 | cut -d/ -f1)
  
  if [ -n "$first_entry" ] && [ -e "$dir/$first_entry" ]; then
    status="SKIP"
    result="OK"
  else
    if tar -xf "$tarfile" -C "$dir" 2>/dev/null; then
      status="EXTRACT"
      result="OK"
    else
      status="EXTRACT"
      result="FAIL"
      [ -n "$first_entry" ] && rm -rf "$dir/$first_entry" 2>/dev/null
    fi
  fi
  
  {
    flock -x 200
    count=$(cat "$counter_file")
    count=$((count + 1))
    echo $count > "$counter_file"
    percent=$((count * 100 / total))
    printf "[%3d%%] (%d/%d) %s-%s: %s\n" $percent $count $total "$status" "$result" "$tarfile"
  } 200>"$counter_file.lock"
'

rm -f "$counter_file" "$counter_file.lock"
echo "Done"