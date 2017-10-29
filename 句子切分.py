#!/usr/bin/python
# -*- coding: UTF-8 -*-

a='"For many prodigies and signs had taken place, and far and wide, over sea and land, the black wings of the Pestilence were spread abroad."'
# 四个分隔符为：,  ;  *  \n
b = ""
flag = False
for i in a:
	if i.isalpha():
		b = b+i
		flag = True
	elif flag:
		print b
		b = ""
		flag = False
		if not i.isspace():
			print i
	elif not i.isspace():
		print i
	# else:
	# 	print("空格")