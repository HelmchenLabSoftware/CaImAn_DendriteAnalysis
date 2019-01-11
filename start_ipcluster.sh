#!/bin/bash

source activate caiman

ipcluster start -n $1
