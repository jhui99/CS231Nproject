import csv

#Column 0 = GEO_ID
#Column 1 = Median Household Income (Estimated)



def readFromFile():
	with open('../DC_GEO_ID_Income_Map.csv') as csv_file:
		GEO_ID_income_dict = {}
		csv_reader = csv.reader(csv_file, delimiter = ',')
		line_count = 0
		for row in csv_reader:
			if line_count != 0:
				GEO_ID = row[0]
				income = row[1]
				GEO_ID_income_dict[GEO_ID] = income
			line_count += 1
	return GEO_ID_income_dict

def writeLabelsToFile(filename, index_label_dict): #not implemented
	f = filename + ".csv"
	with open(f, mode = 'w') as output_file:
		output_writer = csv.writer(output_file, delimiter = ',')
		for index in index_label_dict:
			row = []
			row.append(index)
			row.append(index_label_dict[index])
			output_writer.writerow(row)
##########
		# 		master_list.append(temp)
		# for elem in master_list:
		# 	output_writer.writerow(elem)

def getIncomes():
	with open('../GEO_ID_Income_Map.csv') as csv_file:
		incomes = []
		csv_reader = csv.reader(csv_file, delimiter = ',')
		line_count = 0
		for row in csv_reader:
			if line_count != 0:
				incomes.append(float(row[1]))
			line_count += 1
	return incomes

# counts = getIncomes()
# print(counts[1])