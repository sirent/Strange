
def fcountEO(a):
    odn = 0
    evn = 0
    O = 0
    E = 0
    for i in range(1,a):
        if i % 2 == 0:
            odn = odn + i
            O = O + 1
        else:
            evn = evn + i
            E = E + 1

    print("Sum of odd number: ", odn)
    print("Odd number counter: ", O)
    print("Sum of even number: ", evn)
    print("Even number counter: ", E)

def wcountEO(a,i=1,odn=0,evn=0,E=0,O=0):
    while i <= a:
        if i % 2 == 0:
            odn = odn + i
            O = O + 1
        else:
            evn = evn + i
            E = E + 1
        i+=1

    print("\nSum of odd number: ", odn)
    print("Odd number counter: ", O)
    print("Sum of even number: ", evn)
    print("Even number counter: ", E)


fcountEO(2)
wcountEO(2)

# Ternyata hasilnya berbeda antara for dan while
# Hasil for kehilangan angka terakhir yang seharusnya dihitung tapi tidak ikut terhitung
# Sementara while menghitung range angka hingga akhir
# Untuk mengakali masalah ini kita akan menambah angka inputan dengan 1 sebagai berikut

def fcountEO_new(a):
    odn = 0
    evn = 0
    O = 0
    E = 0
    for i in range(1,a+1):
        if i % 2 == 0:
            odn = odn + i
            O = O + 1
        else:
            evn = evn + i
            E = E + 1

    print("\nSum of odd number: ", odn)
    print("Odd number counter: ", O)
    print("Sum of even number: ", evn)
    print("Even number counter: ", E)

fcountEO_new(2)

# Yak, sudah benar
# Ide ini terinspirasi dari tembok kamar mandi