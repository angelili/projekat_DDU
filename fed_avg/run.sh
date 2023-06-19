#!/bin/bash

# Declare the client job names
clients=("client1" "client2" "client3")



# Loop through the client job names and submit each client job to the "cuda" partition
for client in "${clients[@]}"; do
    sbatch client.sh &
done

# Wait for all background client jobs to finish
wait
#!/bin/bash

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT


