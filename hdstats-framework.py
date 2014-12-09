import dataImport
import dataCleaning
import dataAnalysis


while True:
    s=raw_input('Select task:\n 1. Import\n 2. Data cleaning\n 3. Analysis\n 4. Exit\n')
    if s=='1':
        dataset=dataImport.main()
    elif s=='2':
        dataset=dataCleaning.main(dataset)
    elif s=='3':
        dataset=dataAnalysis.main(dataset)
    elif s=='4':
        break
    else:
        print('Input not recognized')


