#!/bin/bash

source activate caiman

ipcluster start --daemonize -n $1
