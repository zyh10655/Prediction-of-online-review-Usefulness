# Copy data/
scp -P "$1" -r ../data/new_data "$2":/home/data/
echo "Data Transfer complete"

# Copy Python scripts
scp -P "$1" -r ../scripts/*.py "$2":/home/scripts
echo "File Transfer complete"