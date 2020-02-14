
input_file = '/Users/shanglinghsu/OneDrive - HKUST Connect/Data Mining/DataMining/wiki/wiki_en.txt'
output_file = '/Users/shanglinghsu/OneDrive - HKUST Connect/Data Mining/DataMining/wiki/wiki_en_453777.txt'

f = open(output_file,'w')
f.close()

with open(input_file,'r') as f:
    c = 0
    for i,line in enumerate (f):
        if i%10==0:
            c += 1
            o = open(output_file,'a')
            o.write(line)
        if i%10000==0:
            print(i, 'lines appended')
    print(str(c)+" lines in total.")    # mod 10,000 => 454 lines