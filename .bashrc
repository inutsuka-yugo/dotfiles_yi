# To the extent possible under law, the author(s) have dedicated all
# copyright and related and neighboring rights to this software to the
# public domain worldwide. This software is distributed without any warranty.
# You should have received a copy of the CC0 Public Domain Dedication along
# with this software.
# If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.

# base-files version 4.2-4

# ~/.bashrc: executed by bash(1) for interactive shells.

# The latest version as installed by the Cygwin Setup program can
# always be found at /etc/defaults/etc/skel/.bashrc

# Modifying /etc/skel/.bashrc directly will prevent
# setup from updating it.

# The copy in your home directory (~/.bashrc) is yours, please
# feel free to customise it to create a shell
# environment to your liking.  If you feel a change
# would be benifitial to all, please feel free to send
# a patch to the cygwin mailing list.

# User dependent .bashrc file

# If not running interactively, don't do anything
[[ "$-" != *i* ]] && return

# Shell Options
#
# See man bash for more options...
#
# Don't wait for job termination notification
# set -o notify
#
# Don't use ^D to exit
# set -o ignoreeof
#
# Use case-insensitive filename globbing
# shopt -s nocaseglob
#
# Make bash append rather than overwrite the history on disk
# shopt -s histappend
#
# When changing directory small typos can be ignored by bash
# for example, cd /vr/lgo/apaache would find /var/log/apache
# shopt -s cdspell

# Completion options
#
# These completion tuning parameters change the default behavior of bash_completion:
#
# Define to access remotely checked-out files over passwordless ssh for CVS
# COMP_CVS_REMOTE=1
#
# Define to avoid stripping description in --option=description of './configure --help'
# COMP_CONFIGURE_HINTS=1
#
# Define to avoid flattening internal contents of tar files
# COMP_TAR_INTERNAL_PATHS=1
#
# Uncomment to turn on programmable completion enhancements.
# Any completions you add in ~/.bash_completion are sourced last.
# [[ -f /etc/bash_completion ]] && . /etc/bash_completion

# History Options
#
# Don't put duplicate lines in the history.
# export HISTCONTROL=$HISTCONTROL${HISTCONTROL+,}ignoredups
#
# Ignore some controlling instructions
# HISTIGNORE is a colon-delimited list of patterns which should be excluded.
# The '&' is a special pattern which suppresses duplicate entries.
# export HISTIGNORE=$'[ \t]*:&:[fb]g:exit'
# export HISTIGNORE=$'[ \t]*:&:[fb]g:exit:ls' # Ignore the ls command as well
#
# Whenever displaying the prompt, write the previous line to disk
# export PROMPT_COMMAND="history -a"

# Aliases
#
# Some people use a different file for aliases
# if [ -f "${HOME}/.bash_aliases" ]; then
#   source "${HOME}/.bash_aliases"
# fi
#
# Some example alias instructions
# If these are enabled they will be used instead of any instructions
# they may mask.  For example, alias rm='rm -i' will mask the rm
# application.  To override the alias instruction use a \ before, ie
# \rm will call the real rm not the alias.
#
# Interactive operation...
# alias rm='rm -i'
# alias cp='cp -i'
# alias mv='mv -i'
#
# Default to human readable figures
# alias df='df -h'
# alias du='du -h'
#
# Misc :)
# alias less='less -r'                          # raw control characters
# alias whence='type -a'                        # where, of a sort
# alias grep='grep --color'                     # show differences in colour
# alias egrep='egrep --color=auto'              # show differences in colour
# alias fgrep='fgrep --color=auto'              # show differences in colour
#
# Some shortcuts for different directory listings
alias ls='ls -hF -G'                 # classify files in colour
alias dir='ls  --format=vertical'
alias vdir='ls --format=long'
alias ll='ls -l'                              # long list
alias la='ls -A'                              # all but . and ..
alias l='ls -CF'                              #

alias rm='rm -i'	# confirm remove, default is \rm

# Umask
#
# /etc/profile sets 022, removing write perms to group + others.
# Set a more restrictive umask: i.e. no exec perms for others:
# umask 027
# Paranoid: neither group nor others have any perms:
# umask 077

# Functions
#
# Some people use a different file for functions
# if [ -f "${HOME}/.bash_functions" ]; then
#   source "${HOME}/.bash_functions"
# fi
#
# Some example functions:
#
# a) function settitle
# settitle ()
# {
#   echo -ne "\e]2;$@\a\e]1;$@\a";
# }
#
# b) function cd_func
# This function defines a 'cd' replacement function capable of keeping,
# displaying and accessing history of visited directories, up to 10 entries.
# To use it, uncomment it, source this file and try 'cd --'.
# acd_func 1.0.5, 10-nov-2004
# Petar Marinov, http:/geocities.com/h2428, this is public domain
# cd_func ()
# {
#   local x2 the_new_dir adir index
#   local -i cnt
#
#   if [[ $1 ==  "--" ]]; then
#     dirs -v
#     return 0
#   fi
#
#   the_new_dir=$1
#   [[ -z $1 ]] && the_new_dir=$HOME
#
#   if [[ ${the_new_dir:0:1} == '-' ]]; then
#     #
#     # Extract dir N from dirs
#     index=${the_new_dir:1}
#     [[ -z $index ]] && index=1
#     adir=$(dirs +$index)
#     [[ -z $adir ]] && return 1
#     the_new_dir=$adir
#   fi
#
#   #
#   # '~' has to be substituted by ${HOME}
#   [[ ${the_new_dir:0:1} == '~' ]] && the_new_dir="${HOME}${the_new_dir:1}"
#
#   #
#   # Now change to the new dir and add to the top of the stack
#   pushd "${the_new_dir}" > /dev/null
#   [[ $? -ne 0 ]] && return 1
#   the_new_dir=$(pwd)
#
#   #
#   # Trim down everything beyond 11th entry
#   popd -n +11 2>/dev/null 1>/dev/null
#
#   #
#   # Remove any other occurence of this dir, skipping the top of the stack
#   for ((cnt=1; cnt <= 10; cnt++)); do
#     x2=$(dirs +${cnt} 2>/dev/null)
#     [[ $? -ne 0 ]] && return 0
#     [[ ${x2:0:1} == '~' ]] && x2="${HOME}${x2:1}"
#     if [[ "${x2}" == "${the_new_dir}" ]]; then
#       popd -n +$cnt 2>/dev/null 1>/dev/null
#       cnt=cnt-1
#     fi
#   done
#
#   return 0
# }
#
# alias cd=cd_func

# ***************************http://qiita.com/k-takata/items/092f70f66d545cb9db7c******************
# we can see cd history by 'cd --'
cd_func ()
{
      local x2 the_new_dir adir index
        local -i cnt

          if [[ $1 ==  "--" ]]; then
                  dirs -v
                      return 0
                        fi

                          the_new_dir=$1
                            [[ -z $1 ]] && the_new_dir=$HOME

                              if [[ ${the_new_dir:0:1} == '-' ]]; then
                                      #
                                          # Extract dir N from dirs
                                              index=${the_new_dir:1}
                                                  [[ -z $index ]] && index=1
                                                      adir=$(dirs +$index)
                                                          [[ -z $adir ]] && return 1
                                                              the_new_dir=$adir
                                                                fi

                                                                  #
                                                                    # '~' has to be substituted by ${HOME}
                                                                      [[ ${the_new_dir:0:1} == '~' ]] && the_new_dir="${HOME}${the_new_dir:1}"

                                                                        #
                                                                          # Now change to the new dir and add to the top of the stack
                                                                            pushd "${the_new_dir}" > /dev/null
                                                                              [[ $? -ne 0 ]] && return 1
                                                                                the_new_dir=$(pwd)

                                                                                  #
                                                                                    # Trim down everything beyond 11th entry
                                                                                      popd -n +11 2>/dev/null 1>/dev/null

                                                                                        #
                                                                                          # Remove any other occurence of this dir, skipping the top of the stack
                                                                                            for ((cnt=1; cnt <= 10; cnt++)); do
                                                                                                    x2=$(dirs +${cnt} 2>/dev/null)
                                                                                                        [[ $? -ne 0 ]] && return 0
                                                                                                            [[ ${x2:0:1} == '~' ]] && x2="${HOME}${x2:1}"
                                                                                                                if [[ "${x2}" == "${the_new_dir}" ]]; then
                                                                                                                          popd -n +$cnt 2>/dev/null 1>/dev/null
                                                                                                                                cnt=cnt-1
                                                                                                                                    fi
                                                                                                                                      done

                                                                                                                                        return 0
}

alias cd=cd_func
# *************************************************************************************************

alias vi='vim'
export PATH="/mnt/c/Program\ Files/Git\ LFS/:$HOME/.rbenv/bin:$PATH"

alias e='exit'
alias q='exit'
alias bashrc='vi ~/bashrc'
alias brc='bashrc'
alias vimrc='vi ~/.vimrc'
alias vrc='vimrc'
alias py='python3'
alias gnuolotrc='sudo vim /usr/share/gnuplot/gnuplot/5.2/gnuplotrc'
alias gnprc='gnuplotrc'

export DISPLAY="localhost:0.0"

export PS1=" \W>"

alias win='explorer.exe .'

alias ginit='git init; echo "*.a\na\n*.cfg\n~$*" > .gitignore; git add .gitignore; git commit'

alias run='source ~/.*rc'
alias home='cd /mnt/c/Users/yugo'
alias cad='cd /mnt/c/Users/yugo/Documents/ut/robotech/CAD/'
alias 3s='cd /mnt/c/Users/yugo/Documents/ut/study/3s'
alias 3a='cd /mnt/c/Users/yugo/Documents/ut/study/3a'
alias 4s='cd /mnt/c/Users/yugo/Documents/ut/study/4s'
alias 4a='cd /mnt/c/Users/yugo/Documents/ut/study/4a'
alias 5s='cd /mnt/c/Users/yugo/Documents/ut/study/5s'
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
alias lt='latexmk -pvc'
alias c='cd /mnt/c'
alias xming='/mnt/c/Program\ Files\ \(x86\)/Xming/Xming.exe'

#convert
alias conv=convert

alias gnp='"/mnt/c/Program Files (x86)/Xming/Xming.exe" :0 -clipboard -multiwindow& gnuplot'
alias gnup='gnp'
alias tm=tmux

#joke
alias ft=fortune
alias cowsay=cs
alias tux='cs -f tux'
alias gb='cs -f ghostbusters'
alias aa='figlet'
alias mtrx=cmatrix

alias photon="ce05181505@cmp.phys.s.u-tokyo.ac.jp:~"
alias sshp="ssh -X cmp.phys.s.u-tokyo.ac.jp -l ce05181505"

alias jup="win; jupyter notebook"
alias clear="stty sane; clear"

cd ~
tm
fish
