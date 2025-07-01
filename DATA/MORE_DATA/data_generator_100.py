class DataGenerator100:
    def __init__(self):
        self.hta_percentage = 1.0    
        self.hta_essentielle = [] 
        self.hta_secondaire = [] 
        self.complications_hta = [] 

    def get_hta_percentage_of(self, source : list) ->list :
        max_num = len(source)
        percent_num = int(self.hta_percentage * max_num)

        return source[:percent_num]
     
    #region hta (170/320)
    def generate_111(self, number : int)-> list:
        self.hta_essentielle = [
            ("Je me sens souvent fatigué sans raison apparente.",number),
            ("J'ai des maux de tête fréquents, surtout le matin.",number),
            ("Ma vision devient parfois floue sans raison.",number),
            ("Je ressens des étourdissements lorsque je me lève trop vite.",number),
            ("J'ai des saignements de nez occasionnels sans cause évidente.",number),
            ("Je me sens essoufflé même après un effort léger.",number),
            ("J'ai des douleurs thoraciques légères de temps en temps.",number),
            ("Je ressens des palpitations cardiaques sans faire d'exercice.",number),
            ("J'ai des difficultés à dormir et je me réveille souvent la nuit.",number),
            ("Je me sens anxieux et irritable sans raison.",number),
            ("J'ai des bourdonnements d'oreilles fréquents.",number),
            ("Je ressens une pression dans ma poitrine de temps en temps.",number),
            ("J'ai des crampes musculaires sans avoir fait d'effort intense.",number),
            ("Je me sens souvent confus et j'ai du mal à me concentrer.",number),
            ("J'ai des sueurs excessives sans raison apparente.",number),
            ("Je ressens une fatigue intense même après une bonne nuit de sommeil.",number),
            ("J'ai des nausées occasionnelles sans cause identifiable.",number),
            ("Je me sens souvent faible et sans énergie.",number),
            ("J'ai des douleurs dans le cou et les épaules sans raison.",number),
            ("Je ressens une oppression dans la poitrine de temps en temps.",number),
            ("J'ai des vertiges fréquents sans raison apparente.",number),
            ("Je me sens souvent tendu et stressé sans raison.",number),
            ("J'ai des douleurs dans la nuque qui apparaissent soudainement.",number),
            ("Je ressens des picotements dans les mains et les pieds.",number),
            ("J'ai des troubles de la mémoire et j'oublie souvent les choses.",number),
            ("Je me sens souvent étourdi et désorienté.",number),
            ("J'ai des douleurs abdominales sans cause évidente.",number),
            ("Je ressens une fatigue constante même après m'être reposé.",number),
            ("J'ai des maux de tête pulsatiles qui durent longtemps.",number),
            ("Je me sens souvent essoufflé sans avoir fait d'effort.",number),
            ("J'ai des douleurs dans les jambes sans raison apparente.",number),
            ("Je ressens des battements irréguliers dans ma poitrine.",number),
            ("J'ai des troubles du sommeil et je me réveille souvent.",number),
            ("Je me sens souvent agité et incapable de me détendre.",number),
            ("J'ai des douleurs dans la mâchoire sans raison.",number),
            ("Je ressens une pression dans les tempes qui est très inconfortable.",number),
            ("J'ai des troubles de la vision qui viennent et partent.",number),
            ("Je me sens souvent faible et sans force.",number),
            ("J'ai des douleurs dans le dos sans raison apparente.",number),
            ("Je ressens des étourdissements en me levant brusquement.",number),
            ("J'ai des saignements des gencives sans cause évidente.",number),
            ("Je me sens souvent fatigué et sans motivation.",number),
            ("J'ai des douleurs dans les bras sans raison.",number),
            ("Je ressens une oppression dans la tête qui est très désagréable.",number),
            ("J'ai des troubles de l'équilibre sans raison apparente.",number),
            ("Je me sens souvent tendu et incapable de me détendre.",number),
            ("J'ai des douleurs dans les jambes qui me réveillent la nuit.",number),
            ("Je ressens des palpitations qui me font peur.",number),
            ("J'ai des troubles de la concentration qui affectent mon travail.",number),
            ("Je me sens souvent essoufflé même au repos.",number),
            ("J'ai des douleurs dans les épaules sans raison.",number),
            ("Je ressens une fatigue intense qui m'empêche de vaquer à mes occupations.",number),
            ("J'ai des maux de tête qui ne disparaissent pas avec les médicaments.",number),
            ("Je me sens souvent étourdi et j'ai du mal à me concentrer.",number),
            ("J'ai des douleurs dans la poitrine qui me font peur.",number),
            ("Je ressens des picotements dans les doigts sans raison.",number),
            ("J'ai des troubles de la mémoire qui m'inquiètent.",number),
            ("Je me sens souvent faible et sans énergie.",number),
            ("J'ai des douleurs dans le cou qui apparaissent soudainement.",number),
            ("Je ressens des étourdissements fréquents qui m'inquiètent.",number),
            ("J'ai des saignements de nez qui sont très inquiétants.",number),
            ("Je me sens souvent fatigué et sans motivation.",number),
            ("J'ai des douleurs dans les bras sans raison apparente.",number),
            ("Je ressens une oppression dans la tête qui est très inconfortable.",number),
            ("J'ai des troubles de l'équilibre qui m'inquiètent.",number),
            ("Je me sens souvent tendu et incapable de me détendre.",number),
            ("J'ai des douleurs dans les jambes qui me réveillent la nuit.",number),
            ("Je ressens des palpitations qui me font peur.",number),
            ("J'ai des troubles de la concentration qui affectent mon travail.",number),
            ("Je me sens souvent essoufflé même au repos.",number),
            ("J'ai des douleurs dans les épaules sans raison apparente.",number),
            ("Je ressens une fatigue intense qui m'empêche de vaquer à mes occupations.",number),
            ("J'ai des maux de tête qui ne disparaissent pas avec les médicaments.",number),
            ("Je me sens souvent étourdi et j'ai du mal à me concentrer.",number),
            ("J'ai des douleurs dans la poitrine qui me font peur.",number),
            ("Je ressens des picotements dans les doigts sans raison apparente.",number),
            ("J'ai des troubles de la mémoire qui m'inquiètent.",number),
            ("Je me sens souvent faible et sans énergie.",number),
            ("J'ai des douleurs dans le cou qui apparaissent soudainement.",number),
            ("Je ressens des étourdissements fréquents qui m'inquiètent.",number),
            ("J'ai des saignements de nez qui sont très inquiétants.",number),
            ("Je me sens souvent fatigué et sans motivation.",number),
            ("J'ai des douleurs dans les bras sans raison apparente.",number),
            ("Je ressens une oppression dans la tête qui est très inconfortable.",number),
            ("J'ai des troubles de l'équilibre qui m'inquiètent.",number),
            ("Je me sens souvent tendu et incapable de me détendre.",number),
            ("J'ai des douleurs dans les jambes qui me réveillent la nuit.",number),
            ("Je ressens des palpitations qui me font peur.",number),
            ("J'ai des troubles de la concentration qui affectent mon travail.",number),
            ("Je me sens souvent essoufflé même au repos.",number),
            ("J'ai des douleurs dans les épaules sans raison apparente.",number),
            ("Je ressens une fatigue intense qui m'empêche de vaquer à mes occupations.",number),
            ("J'ai des maux de tête qui ne disparaissent pas avec les médicaments.",number),
            ("Je me sens souvent étourdi et j'ai du mal à me concentrer.",number),
            ("J'ai des douleurs dans la poitrine qui me font peur.",number),
            ("Je ressens des picotements dans les doigts sans raison apparente.",number),
            ("J'ai des troubles de la mémoire qui m'inquiètent.",number),
            ("Je me sens souvent faible et sans énergie.",number),
            ("J'ai des douleurs dans le cou qui apparaissent soudainement.",number),
            ("Je ressens des étourdissements fréquents qui m'inquiètent.",number),
            ("J'ai des saignements de nez qui sont très inquiétants.",number),
            ("Je me sens souvent fatigué et sans motivation.",number),
            ("J'ai des douleurs dans les bras sans raison apparente.",number),
            ("Je ressens une oppression dans la tête qui est très inconfortable.",number),
            ("J'ai des troubles de l'équilibre qui m'inquiètent.",number),
            ("Je me sens souvent tendu et incapable de me détendre.",number),
            ("J'ai des douleurs dans les jambes qui me réveillent la nuit.",number),
            ("Je ressens des palpitations qui me font peur.",number),
            ("J'ai des troubles de la concentration qui affectent mon travail.",number),
            ("Je me sens souvent essoufflé même au repos.",number),
            ("J'ai des douleurs dans les épaules sans raison apparente.",number),
            ("Je ressens une fatigue intense qui m'empêche de vaquer à mes occupations.",number),
            ("J'ai des maux de tête qui ne disparaissent pas avec les médicaments.",number),
            ("Je me sens souvent étourdi et j'ai du mal à me concentrer.",number),
            ("J'ai des douleurs dans la poitrine qui me font peur.",number),
            ("Je ressens des picotements dans les doigts sans raison apparente.",number),
            ("J'ai des troubles de la mémoire qui m'inquiètent.",number),
            ("Je me sens souvent faible et sans énergie.",number),
            ("J'ai des douleurs dans le cou qui apparaissent soudainement.",number),
            ("Je ressens des étourdissements fréquents qui m'inquiètent.",number),
            ("J'ai des saignements de nez qui sont très inquiétants.",number),
            ("Je me sens souvent fatigué et sans motivation.",number),
            ("J'ai des douleurs dans les bras sans raison apparente.",number),
            ("Je ressens une oppression dans la tête qui est très inconfortable.",number),
            ("J'ai des troubles de l'équilibre qui m'inquiètent.",number),
            ("Je me sens souvent tendu et incapable de me détendre.",number),
            ("J'ai des douleurs dans les jambes qui me réveillent la nuit.",number),
            ("Je ressens des palpitations qui me font peur.",number),
            ("J'ai des troubles de la concentration qui affectent mon travail.",number),
            ("Je me sens souvent essoufflé même au repos.",number),
            ("J'ai des douleurs dans les épaules sans raison apparente.",number),
            ("Je ressens une fatigue intense qui m'empêche de vaquer à mes occupations.",number),
            ("J'ai des maux de tête qui ne disparaissent pas avec les médicaments.",number),
            ("Je me sens souvent étourdi et j'ai du mal à me concentrer.",number),
            ("J'ai des douleurs dans la poitrine qui me font peur.",number),
            ("Je ressens des picotements dans les doigts sans raison apparente.",number),
            ("J'ai des troubles de la mémoire qui m'inquiètent.",number),
            ("Je me sens souvent faible et sans énergie.",number),
            ("J'ai des douleurs dans le cou qui apparaissent soudainement.",number),
            ("Je ressens des étourdissements fréquents qui m'inquiètent.",number),
            ("J'ai des saignements de nez qui sont très inquiétants.",number),
            ("Je me sens souvent fatigué et sans motivation.",number),
            ("J'ai des douleurs dans les bras sans raison apparente.",number),
            ("Je ressens une oppression dans la tête qui est très inconfortable.", number)
        ]

        return self.get_hta_percentage_of(self.hta_essentielle)

    def generate_112(self, number:int)-> list:
        self.hta_secondaire = [
            ("J'ai remarqué que j'ai souvent mal à la tête et que ma vision devient floue de temps en temps.",number),
            ("Je me sens constamment fatigué et j'ai des difficultés à respirer même après un effort léger.",number),
            ("Mon médecin m'a dit que j'avais un bruit dans les artères du cou lors de mon dernier examen.",number),
            ("J'ai des palpitations fréquentes et je me sens souvent anxieux sans raison apparente.",number),
            ("J'ai pris du poids récemment et j'ai remarqué que je transpire beaucoup plus qu'avant.",number),
            ("Je ressens souvent des étourdissements et j'ai parfois des saignements de nez sans raison.",number),
            ("J'ai des douleurs dans la poitrine et mon médecin a mentionné que ma pression artérielle était très élevée.",number),
            ("J'ai des crampes musculaires fréquentes et une sensation de faiblesse générale dans mon corps.",number),
            ("J'ai des difficultés à dormir et je me réveille souvent la nuit avec une sensation d'oppression.",number)
        ]   

        return self.get_hta_percentage_of(self.hta_secondaire)
    
    def generate_113(self, number : int)-> list:
        self.complications_hta = [
            ("Je ressens une douleur intense dans la poitrine qui irradie vers mon bras gauche.",number), 
            ("J'ai souvent des essoufflements même lorsque je suis au repos.",number), 
            ("Je me sens très fatigué et faible, même après une bonne nuit de sommeil.",number), 
            ("J'ai des palpitations cardiaques qui me font très peur.",number), 
            ("Mes jambes gonflent et je ressens une lourdeur constante.",number), 
            ("Je tousse beaucoup la nuit et j'ai du mal à respirer quand je suis allongé.",number), 
            ("J'ai des étourdissements fréquents et je me sens parfois proche de l'évanouissement.",number), 
            ("Ma tension artérielle reste élevée malgré mes médicaments.,",number), 
            ("Je ressens une oppression dans la poitrine qui ne disparaît pas.",number), 
            ("J'ai des maux de tête violents qui ne s'améliorent pas avec les analgésiques.",number), 
            ("Je me réveille souvent la nuit avec une sensation d'étouffement.",number), 
            ("J'ai des douleurs dans le dos qui ne sont pas soulagées par le repos.",number), 
            ("Je ressens une grande anxiété et j'ai peur de faire un effort physique.",number), 
            ("J'ai des nausées et des sueurs froides sans raison apparente.",number), 
            ("Je ressens des battements irréguliers dans ma poitrine.",number), 
            ("J'ai des difficultés à marcher en raison d'une douleur dans la jambe.",number), 
            ("Je me sens confus et j'ai du mal à me concentrer sur des tâches simples.",number)
        ]   

        return self.get_hta_percentage_of(self.complications_hta)
    

    #endregion hta (170/320)

    #region diabete (85/320)
    def generate_121(self, number:int)-> list:
        diabete_type2 = [

            ("J'ai souvent soif même après avoir bu beaucoup d'eau.",number),
            ("Je me sens très fatigué ces derniers temps sans raison.",number),
            ("J'urine beaucoup plus souvent surtout la nuit.",number),
            ("J'ai remarqué que mes plaies guérissent très lentement.",number),
            ("Mes pieds me picotent et parfois j'ai des fourmillements.",number),
            ("J'ai perdu du poids sans faire de régime.",number),
            ("J'ai souvent faim même après les repas.",number),
            ("Ma vision est parfois floue surtout en fin de journée.",number),
            ("J'ai une sensation de brûlure dans mes jambes.",number),
            ("Je me sens irritable sans raison apparente.",number),
            ("J'ai des infections urinaires à répétition.",number),
            ("J'ai des démangeaisons fréquentes au niveau de la peau.",number),
            ("J'ai la bouche sèche en permanence.",number),
            ("Je remarque que mes gencives saignent facilement.",number),
            ("J'ai des crampes musculaires qui reviennent souvent.",number),
            ("Je me sens faible et sans énergie.",number),
            ("J'ai des vertiges lorsque je me lève trop vite.",number),
            ("J'ai du mal à cicatriser après une coupure.",number),
            ("Mes pieds sont froids et j'ai parfois des douleurs.",number),
            ("J'ai des infections de la peau qui ne passent pas facilement.",number),
            ("Mon appétit est très variable je peux manger beaucoup puis rien.",number),
            ("Je transpire beaucoup même sans effort.",number),
            ("J'ai souvent des nausées sans cause évidente.",number),
            ("Mes mains sont souvent engourdies.",number),
            ("J'ai des picotements dans les doigts et les orteils.",number),
            ("Je perds l'équilibre de temps en temps.",number),
            ("J'ai du mal à me concentrer au travail.",number),
            ("Je me sens parfois très anxieux sans raison.",number),
            ("J'ai remarqué que ma peau est devenue plus foncée au niveau du cou.",number),
            ("J'ai une sensation de soif intense la nuit.",number),
            ("J'ai souvent la tête lourde et mal au crâne.",number),
            ("J'ai un goût métallique dans la bouche.",number),
            ("Mes ongles sont cassants et déformés.",number),
            ("Je ressens une fatigue intense après chaque repas.",number),
            ("J'ai du mal à rester debout longtemps mes jambes me font mal.",number),
            ("J'ai remarqué que mes vêtements sont plus larges j'ai perdu du ventre.",number),
            ("Je suis souvent en hypoglycémie j'ai des sueurs froides.",number),
            ("J'ai des troubles du sommeil je me réveille souvent la nuit.",number),
            ("J'ai la peau très sèche surtout sur les jambes.",number),
            ("Mes lèvres sont souvent gercées et douloureuses.",number),
            ("J'ai des douleurs dans la poitrine quand je marche longtemps.",number),
            ("Je ressens une grande soif qui ne passe pas.",number),
            ("J'ai des nausées après les repas gras.",number),
            ("Mes pieds ont des rougeurs et des gonflements.",number),
            ("Je ressens une faiblesse soudaine en faisant de l'effort.",number),
            ("J'ai des douleurs articulaires fréquentes.",number),
            ("J'ai du mal à respirer quand je fais une petite marche.",number),
            ("J'ai des maux de tête qui reviennent chaque jour.",number),
            ("Je me sens souvent déprimé.",number),
            ("J'ai des brûlures d'estomac récurrentes.",number),
            ("Mes yeux piquent et je pleure facilement.",number),
            ("J'ai une sensation de lourdeur dans les jambes.",number),
            ("J'ai des sensations de picotements quand je conduis.",number),
            ("J'ai remarqué que mes cheveux tombent plus qu'avant.",number),
            ("J'ai souvent froid même quand il fait chaud.",number),
            ("Je ressens une grande faiblesse après le sport.",number),
            ("J'ai des troubles de la mémoire récents.",number),
            ("Je remarque que mes gencives sont rouges et enflées.",number),
            ("J'ai des démangeaisons au niveau des plis de la peau.",number),
            ("J'ai des crampes nocturnes dans les mollets.",number),
            ("Je suis souvent essoufflé en montant les escaliers.",number),
            ("J'ai une mauvaise haleine qui ne part pas.",number),
            ("Je suis souvent irritable et de mauvaise humeur.",number),
            ("J'ai des infections vaginales à répétition.",number),
            ("Je ressens des douleurs au ventre après les repas.",number),
            ("J'ai des troubles digestifs fréquents.",number),
            ("J'ai du mal à perdre du poids malgré mes efforts.",number),
            ("Mes pieds sont souvent engourdis après avoir marché.",number),
            ("J'ai des difficultés à me concentrer quand je travaille sur l'ordinateur.",number),
            ("J'ai souvent les mains froides et moites.",number),
            ("Je ressens une fatigue chronique qui ne s'améliore pas avec le repos.",number),
            ("J'ai remarqué que ma peau est devenue plus fine.",number),
            ("Je ressens une grande soif et une bouche pâteuse.",number),
            ("J'ai souvent des palpitations sans cause apparente.",number),
            ("J'ai du mal à contrôler ma glycémie malgré mon traitement.",number)        
        ]
        return diabete_type2

    def generate_122(self, number:int)-> list:
        diabete_type1 = [
            ("Je me sens souvent très fatigué, même après une bonne nuit de sommeil.",number),
            ("J'ai une soif intense et je dois boire de l'eau constamment.",number),
            ("Je vais aux toilettes très fréquemment, même pendant la nuit.",number),
            ("J'ai perdu du poids sans raison apparente, malgré un appétit normal.",number),
            ("J'ai souvent des étourdissements et je me sens faible.",number),
            ("Ma vision est devenue floue et j'ai du mal à me concentrer.",number),
            ("Je ressens des picotements et un engourdissement dans mes mains et mes pieds.",number),
            ("J'ai des infections fréquentes, comme des infections de la peau ou des infections urinaires.",number),
            ("Je me sens souvent irritable et anxieux sans raison évidente.",number),
            ("J'ai des épisodes de transpiration excessive et des tremblements, surtout si je n'ai pas mangé à temps.",number)
        ]
        return diabete_type1
    
    #endregion diabete (85/320)

    #region troubles du sommeil et psychiques (50/320)

    def generate_131(self, number:int)-> list:
        depression_majeure = [
            ("Je me sens constamment triste et vide, sans raison apparente.",number),
            ("Je n'ai plus d'intérêt pour les activités que j'aimais auparavant.",number),
            ("Je me réveille souvent la nuit et j'ai du mal à me rendormir.",number),
            ("Je me sens fatigué tout le temps, même après une bonne nuit de sommeil.",number),
            ("J'ai du mal à me concentrer, même sur des tâches simples.",number),
            ("Je me sens inutile et sans espoir pour l'avenir.",number),
            ("J'ai perdu l'appétit et j'ai du mal à manger régulièrement.",number),
            ("Je me sens coupable et je rumine souvent sur mes erreurs passées.",number),
            ("Je n'arrive plus à prendre des décisions, même les plus simples.",number),
            ("Je me sens agité et incapable de rester en place.",number),
            ("Je pense souvent à la mort et à la possibilité de me suicider.",number),
            ("Je pleure souvent sans raison apparente.",number),
            ("Je me sens physiquement ralenti, comme si tout était un effort.",number),
            ("Je n'ai plus d'énergie pour faire quoi que ce soit.",number),
            ("Je me sens désespéré et je ne vois pas d'issue à ma situation.",number),
            ("Je me sens irritable et je m'énerve facilement.",number),
            ("Je n'arrive plus à ressentir de la joie ou du plaisir.",number),
            ("Je me sens submergé par l'anxiété et la peur.",number),
            ("Je me sens comme un fardeau pour les autres.",number),
            ("Je n'ai plus envie de sortir de chez moi ou de voir qui que ce soit.",number),
            ("Je me sens vide et sans émotion.",number),
            ("Je n'arrive plus à me motiver pour quoi que ce soit.",number),
            ("Je me sens constamment en échec.",number),
            ("Je n'ai plus confiance en moi et je me sens inutile.",number),
            ("Je me sens coupable de ne pas être à la hauteur.",number),
            ("Je n'arrive plus à me projeter dans l'avenir.",number),
            ("Je me sens comme si je vivais dans un brouillard.",number),
            ("Je n'ai plus envie de prendre soin de moi.",number),
            ("Je me sens comme si je perdais le contrôle de ma vie.",number),
            ("Je n'arrive plus à ressentir de l'affection pour les autres.",number),
            ("Je me sens comme si je n'avais plus rien à offrir.",number),
            ("Je n'ai plus envie de parler à qui que ce soit.",number),
            ("Je me sens comme si je n'avais plus de raison de vivre.",number),
            ("Je n'arrive plus à me concentrer sur quoi que ce soit.",number),
            ("Je me sens comme si je n'avais plus de valeur.",number)
        ]
        return depression_majeure

    def generate_132(self, number:int)-> list:
        anxiete_generalisee = [
            ("Je me sens constamment tendu et je m'inquiète pour tout, même les petites choses.",number),
            ("Mon esprit ne s'arrête jamais, c'est comme si je ne pouvais pas contrôler mes pensées.",number),
            ("J'ai souvent des maux de tête et des tensions musculaires sans raison apparente.",number),
            ("Je me sens fatigué tout le temps, même après une bonne nuit de sommeil.",number),
            ("J'ai des difficultés à me concentrer sur mon travail ou mes études à cause de mon anxiété.",number),
            ("Je ressens une peur intense et irrationnelle que quelque chose de terrible va arriver.,number",number),
            ("Je me réveille souvent la nuit en sueur, le cœur battant à cause de mes inquiétudes.",number),
            ("Je me sens submergé par mes responsabilités et j'ai peur de ne pas être à la hauteur.",number),
            ("J'ai des nausées et des problèmes digestifs quand je suis particulièrement anxieux.",number),
            ("Je me sens irritable et à fleur de peau, même avec mes proches.",number),
            ("J'ai souvent des vertiges et je me sens étourdi sans raison médicale évidente.",number),
            ("Je me sens comme si je perdais le contrôle de ma vie et de mes émotions.",number),
            ("J'ai des difficultés à prendre des décisions, même simples, par peur de faire une erreur.",number),
            ("Je me sens souvent en danger, même dans des situations parfaitement sûres.",number),
            ("J'ai des palpitations et une sensation d'oppression dans la poitrine quand je suis anxieux.",number),
            ("Je me sens isolé et incompris, comme si personne ne pouvait comprendre ce que je vis.",number),
            ("J'ai des difficultés à respirer correctement quand je suis submergé par l'anxiété.",number),
            ("Je me sens constamment sur le qui-vive, comme si je devais me préparer au pire.",number),
            ("J'ai des tremblements et des secousses musculaires quand je suis particulièrement stressé.",number),
            ("Je me sens désespéré et sans espoir, comme si rien ne pouvait aller mieux.",number) 
        ]
        return anxiete_generalisee 

    def generate_133(self, number:int)-> list:
        troubles_bipolaires = [
            ("Parfois, je me sens invincible et plein d'énergie, mais d'autres fois, je ne peux même pas sortir du lit.",number),
            ("J'ai des périodes où je parle très vite et où mes pensées s'emballent, puis soudain, je me sens complètement vide.",number),
            ("Il m'arrive de dépenser beaucoup d'argent de manière impulsive quand je me sens au sommet de ma forme.",number),
            ("Je passe par des phases où je me sens extrêmement irritable et agité sans raison apparente.",number),
            ("Il y a des moments où je me sens tellement triste et désespéré que je ne vois pas l'intérêt de continuer.",number)
        ]
        return troubles_bipolaires
 

    def generate_134(self, number:int)-> list:
        troubles_du_sommeil_associes = [
            ("Je me réveille plusieurs fois par nuit et j'ai du mal à me rendormir, ce qui me laisse épuisé le matin.",number),
            ("Même si je me sens fatigué, je ne parviens pas à m'endormir, mon esprit ne cesse de ressasser des pensées.",number),
            ("Je fais souvent des cauchemars qui me réveillent en sueur et en état de panique.",number),
            ("Je me sens comme paralysé au moment de m'endormir ou de me réveiller, c'est vraiment effrayant.",number),
            ("Je passe des nuits blanches à fixer le plafond, et le manque de sommeil aggrave mon anxiété pendant la journée.",number)
        ]
        return troubles_du_sommeil_associes
    
    #endregion troubles du sommeil et psychiques (50/320)

    #endregion 320