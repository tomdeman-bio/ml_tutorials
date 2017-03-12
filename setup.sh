!#/usr/bin/env bash

case "$(uname -s)" in

	Darwin)
        	echo 'Mac OS X'
		wget https://repo.continuum.io/archive/Anaconda2-4.3.1-MacOSX-x86_64.sh
		chmod u+x Anaconda2-4.3.1-MacOSX-x86_64.sh
		./Anaconda2-4.3.1-MacOSX-x86_64.sh
		;;
	Linux)
		echo 'Linux'
		wget https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh
		chmod u+x Anaconda2-4.3.1-Linux-x86_64.sh
		./Anaconda2-4.3.1-Linux-x86_64.sh -b
		;;

	CYGWIN*|MINGW32*|MSYS*)
		echo 'MS Windows'
		echo 'Instructions: '
		;;
	*)
		echo 'can not determine OS' 
		;;
esac

