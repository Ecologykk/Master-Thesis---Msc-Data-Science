def extract_middle_part_of_url(url):
    parts = url.split('https://www.dgsi.pt')[1].split('Pesquisa+Livre?OpenForm')[0]
    return str(parts)

#Testes rápidos
# url = "https://www.dgsi.pt/jtre.nsf/Pesquisa+Livre?OpenForm"
# result = extract_middle_part_of_url(url)
# print(result)  # Output: jtrl.nsf