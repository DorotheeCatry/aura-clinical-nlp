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
        # Ajout d'un mapping pour les noms des pathologies les plus courantes
        # 110 : hypertension artérielle
        self.pathology_mapping[111]= "Hypertension artérielle : HTA essentielle",
        self.pathology_mapping[112]= "Hypertension artérielle : HTA secondaire", 
        self.pathology_mapping[113]= "Hypertension artérielle : Complications cardiovasculaires de l'HTA"

        self.all_data = self.data100.generate_111()
        self.all_data.extend(self.data100.generate_112())
        self.all_data.extend(self.data100.generate_113())

        # 120 : Diabete
        self.pathology_mapping[121] = "Diabète de type 2" # plus fréquent donc en premier
        self.pathology_mapping[122] = "Diabète de type 1"

        self.all_data.extend(self.data100.generate_121())
        self.all_data.extend(self.data100.generate_122())

        # 130 : Troubles de l'humeur et psychiques
        self.pathology_mapping[131] = "Troubles de l'humeur et psychiques : Dépression majeure"
        self.pathology_mapping[132] = "Troubles de l'humeur et psychiques : Anxiété généralisée"
        self.pathology_mapping[133] = "Troubles de l'humeur et psychiques : Troubles bipolaires "
        self.pathology_mapping[134] = "Troubles de l'humeur et psychiques : Troubles du sommeil associés"

        self.all_data.extend(self.data100.generate_131())
        self.all_data.extend(self.data100.generate_132())
        self.all_data.extend(self.data100.generate_133())
        self.all_data.extend(self.data100.generate_134())

        # Création du DataFrame
        df_medical = pd.DataFrame(self.all_data, columns=['phrase', 'pathologie'])
        df_medical['nom_pathologie'] = df_medical['pathologie'].map(self.pathology_mapping)

        return df_medical


    