f = open('./test_stg2.csv', 'w')
for i in range(0, 3506):
    f.write('test_stg2/test_stg2/{}.jpg,,\n'.format(10000 + i)) # 10000.jpg ~ 13505.jpg
