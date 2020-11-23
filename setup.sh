#!/bin/bash

DOT_FILES=(.bashrc .vimrc .gitconfig .tmux.conf)

for file in ${DOT_FILES[@]}
	do
		ln -s $HOME/dotfiles_yi/$file $HOME/$file
	done

ln -s $HOME/dotfiles_yi/config.fish $HOME/.config/fish
ln -s $HOME/dotfiles_yi/custom.js $HOME/.jupyter/custom
ln -s /mnt/c/Users/yugo/Documents
