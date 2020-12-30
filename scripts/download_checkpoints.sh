#!/bin/bash

FACADES_LINK="https://www.dropbox.com/s/han7rizgz6p93d1/best_checkpoint_facades.pth"
EDGES_LINK="https://www.dropbox.com/s/8mf8790dcuyjv81/best_checkpoint_edges2shoes.pth"

PREFIX="../logs/checkpoints"
FACADES_PATH="$PREFIX/best_checkpoint_facades.pth"
EDGES_PATH="$PREFIX/best_checkpoint_edges.pth"

if [ $# -eq 1 ]; then
    DATASET="$1"
else
    DATASET="both"
fi

if [ $DATASET = "facades" ] || [ $DATASET = "both" ]; then
    if [ ! -f $FACADES_PATH ]; then
        echo "Loading facades checkpoint..."
        wget $FACADES_LINK
        mv "best_checkpoint_facades.pth" $FACADES_PATH
        echo "Facades checkpoint has been successfully downloaded."
    else
        echo "Facades checkpoint is already downloaded."
    fi
fi

if [ $DATASET = "edges" ] || [ $DATASET = "both" ]; then
    if [ ! -f $EDGES_PATH ]; then
        echo "Edges facades checkpoint..."
        wget $EDGES_LINK
        mv "best_checkpoint_edges.pth" $EDGES_PATH
        echo "Edges checkpoint has been successfully downloaded."
    else
        echo "Edges checkpoint is already downloaded."
    fi
fi
