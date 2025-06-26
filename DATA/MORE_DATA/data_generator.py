import pandas as pd

from data_generator_100 import DataGenerator100
#from data_generator_200 import DataGenerator200

class DataGenerator:
    def __init__(self):
        self.pathology_mapping = {}
        self.all_data = []
        self.data100 = DataGenerator100()
        #self.data200 = DataGenerator200()

    def get_df_medical(self) -> pd.DataFrame:
        self.pathology_mapping = {}
        self.all_data = []
        # Ajout d'un mapping pour les noms des pathologies les plus courantes
        # 110 : hypertension artérielle

        #self.pathology_mapping[0]= "Hypertension artérielle : HTA essentielle",
        #self.pathology_mapping[1]= "Hypertension artérielle : HTA secondaire", 
        #self.pathology_mapping[2]= "Hypertension artérielle : Complications cardiovasculaires de l'HTA"

        self.pathology_mapping[0]= "Hypertension artérielle",


        self.all_data = self.data100.generate_111(0)
        self.all_data.extend(self.data100.generate_112(0))
        self.all_data.extend(self.data100.generate_113(0))

        # 120 : Diabete
        # self.pathology_mapping[3] = "Diabète de type 2" # plus fréquent donc en premier
        # self.pathology_mapping[4] = "Diabète de type 1"

        self.pathology_mapping[2] = "Diabète"

        self.all_data.extend(self.data100.generate_121(1))
        self.all_data.extend(self.data100.generate_122(1))

        # 130 : Troubles de l'humeur et psychiques
        # self.pathology_mapping[5] = "Troubles de l'humeur et psychiques : Dépression majeure"
        # self.pathology_mapping[6] = "Troubles de l'humeur et psychiques : Anxiété généralisée"
        # self.pathology_mapping[7] = "Troubles de l'humeur et psychiques : Troubles bipolaires "
        # self.pathology_mapping[8] = "Troubles de l'humeur et psychiques : Troubles du sommeil associés"

        self.pathology_mapping[3] = "Troubles psychiques"

        self.all_data.extend(self.data100.generate_131(2))
        self.all_data.extend(self.data100.generate_132(2))
        self.all_data.extend(self.data100.generate_133(2))
        self.all_data.extend(self.data100.generate_134(2))

        # Création du DataFrame
        df_medical = pd.DataFrame(self.all_data, columns=['text', 'label'])
        #df_medical['nom_pathologie'] = df_medical['pathologie'].map(self.pathology_mapping)

        return df_medical


    