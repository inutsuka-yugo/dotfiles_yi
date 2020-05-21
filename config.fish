# Fish git prompt
set __fish_git_prompt_showdirtystate 'yes'
set __fish_git_prompt_showstashstate 'yes'
set __fish_git_prompt_showuntrackedfiles 'yes'
set __fish_git_prompt_showupstream 'yes'
set __fish_git_prompt_color_branch yellow
set __fish_git_prompt_color_upstream_ahead green
set __fish_git_prompt_color_upstream_behind red

# Status Chars
set __fish_git_prompt_char_dirtystate '⚡'
set __fish_git_prompt_char_stagedstate '￫'
set __fish_git_prompt_char_untrackedfiles '='
set __fish_git_prompt_char_stashstate '↩'
set __fish_git_prompt_char_upstream_ahead '+'
set __fish_git_prompt_char_upstream_behind '-'

# Some shortcuts for different directory listings
alias ls='ls -hF -G'                 # classify files in colour
alias dir='ls  --format=vertical'
alias vdir='ls --format=long'
alias ll='ls -l'                              # long list
alias la='ls -A'                              # all but . and ..
alias l='ls -CF'                              #
alias rm='rm -i'    # confirm remove

alias g++='g++ -std=c++11 -Wall -Wextra -Wconversion'

# show git branch name
function git_branch
	git branch --no-color 2> /dev/null | sed -e '/^[^*]/d'
end

# *** alias for stack ***
alias ghc="stack ghc"
alias ghci="stack ghci"
alias runghc="stack runghc"
alias runhaskell="stack runhaskell"
# ******

alias haskell='cd /mnt/c/Users/yugo/Documents/programing/haskell/'
alias hsk='haskell'
alias e='exit;stty sane'
alias q='exit;stty sane'
alias bashrc='vim ~/.bashrc'
alias brc='bashrc'
alias vimrc='vim ~/.vimrc'
alias vrc='vimrc'
alias config='vim ~/.config/fish/config.fish'
alias fishconfig='config'
alias fishfunc='vim ~/.config/fish/functions/fish_prompt.fish'
alias fishrc='config'
alias ffunc='fishfunc'
alias frc='fishrc'
alias py='python'
alias gnuolotrc='sudo vim /usr/share/gnuplot/gnuplot/5.2/gnuplotrc'
alias gnprc=gnuplotrc
alias showpath='python -c "import os; print(\"\n\".join(os.environ[\"PATH\"].split(\" \")))"'

alias search='find ./ -type f -name'

set -x DISPLAY localhost:0.0

# eval 'dircolors ~/.colorrc -b'
#↑なぜかうまくいかない

alias win='explorer.exe .'

#alias init='init && echo "*.a\na\n*.cfg" > .gitignore'

alias frun='source ~/.config/fish/config.fish'
alias home='cd /mnt/c/Users/yugo'
alias cad='cd /mnt/c/Users/yugo/Documents/CR/CAD/'
alias 3s='cd /mnt/c/Users/yugo/Documents/study/3s'
alias 3a='cd /mnt/c/Users/yugo/Documents/study/3a'
alias 4s='cd /mnt/c/Users/yugo/Documents/study/4s'
alias 4a='cd /mnt/c/Users/yugo/Documents/study/4a'
alias 5s='cd /mnt/c/Users/yugo/Documents/study/5s'
alias yok='cd /mnt/c/Users/yugo/Documents/study/okada_lab'
alias cr='cd /mnt/c/Users/yugo/Documents/CR'

alias g='git'
alias gitignore='vim .gitignore'
alias gi='gitignore'

alias ud='sudo apt update'
alias ug='sudo apt upgrade -y'
alias autorm='sudo apt autoremove -y'
alias ins='sudo apt install'
alias unins='sudo apt remove'

alias desk='cd /mnt/c/Users/yugo/Desktop'
alias doc='cd /mnt/c/Users/yugo/Documents'
alias pic='cd /mnt/c/Users/yugo/Pictures'
alias pr='cd /mnt/c/Users/yugo/Pictures/robotech/'
alias tx='cd /mnt/c/Users/yugo/Documents/tex;'
alias lt='latexmk -pvc'
alias c='cd /mnt/c'
alias xming='Xming.exe'

alias xelatex='xelatex.exe'

#convert
alias conv=convert

alias gnp='"/mnt/c/Program Files (x86)/Xming/Xming.exe" :0 -clipboard -multiwindow& gnuplot'
alias gnup='gnuplot'
alias tm=tmux

alias firefox='/mnt/c/Program\ Files/Mozilla\ Firefox/firefox.exe'

#robotech
alias sshar='ssh robotech@10.10.10.23'
alias sshmr='ssh robotech@10.10.10.21'

#joke
alias ft=fortune 
alias cowsay=cs
alias tux='cs -f tux'
alias gb='cs -f ghostbusters'
alias aa='figlet'
alias mtrx=cmatrix

alias photon="ce05181505@cmp.phys.s.u-tokyo.ac.jp:~"
alias sshp="ssh -X cmp.phys.s.u-tokyo.ac.jp -l ce05181505"

alias jup="tm split-window -v -c $PWD; firefox; jupyter notebook"

alias sshdobot="ssh kemako@192.168.10.108"
alias sshker="ssh ubuntu@3.16.89.76 -L 8888:localhost:8888 -p 2211"

#fzf
#set -U FZF_LEGACY_KEYBINDINGS 0

#pyenv
#set -x PYENV_ROOT $HOME/.pyenv
#set -x PATH $PYENV_ROOT/bin:$PATH
#set -x PYTHONPATH ./.pyenv/versions/anaconda3-5.2.0/envs/physics/lib/python3.7/site-packages/iminuit:$PYTHONPATH
#eval (pyenv init - | source)

#conda
#source (conda info --root)/etc/fish/conf.d/conda.fish

#conda activate physics

#bobthefish https://stackoverflow.com/questions/52297324/bobthefish-no-longer-displaying-correctly-for-mercurial-works-fine-for-git/52300013
set -g theme_display_git_ahead_verbose yes 
set -g theme_display_git_dirty_verbose yes 
set -g theme_display_git_master_branch yes 
set -g theme_git_worktree_support yes 
set -g theme_display_hg yes 
set -g theme_display_user ssh 
set -g theme_display_hostname ssh 
set -g theme_display_cmd_duration yes 
set -g theme_title_display_process yes 
set -g theme_title_display_user yes 
set -g theme_title_use_abbreviated_path no 
set -g theme_date_format "+%Y-%m-%d %H:%M" 
set -g theme_avoid_ambiguous_glyphs yes 
set -g theme_show_exit_status yes 
set -g default_user xxx 
set -g theme_project_dir_length 1 
set -g theme_newline_cursor yes

