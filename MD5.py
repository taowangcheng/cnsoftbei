import hashlib

a = 'U20201083220020119'
md5 = hashlib.md5()
md5.update(a.encode('utf-8'))
print(md5.hexdigest())


