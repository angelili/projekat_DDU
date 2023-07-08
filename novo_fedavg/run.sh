#!/bin/bash

# Declare the client job names and corresponding scripts
clients=("client1" "client2" "client3" "client4")
scripts=("client_1.sh" "client_2.sh" "client_3.sh" "client_4.sh")

# Loop through the client job names and submit each client job
for ((i=0; i<${#clients[@]}; i++)); do
    client=${clients[i]}
    script=${scripts[i]}
    sbatch $script &
done



# Wait for all background client jobs to finish
wait


# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT


