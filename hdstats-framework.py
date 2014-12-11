import dataImport
import dataCleaning
import dataAnalysis


while True:
    s=raw_input('Select task:\n 1. Import\n 2. Data cleaning\n 3. Analysis\n 0. Exit\n')
    if s=='0':
        break
    elif s=='1':
        dataset=dataImport.main()
    elif s=='2':
        dataset=dataCleaning.main(dataset)
    elif s=='3':
        dataset=dataAnalysis.main(dataset)

    else:
        print('Input not recognized')


