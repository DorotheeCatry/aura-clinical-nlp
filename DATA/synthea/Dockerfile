FROM openjdk:11

# Installer git et unzip
RUN apt-get update && apt-get install -y git unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copier le code synthea dans le conteneur
COPY . /synthea
WORKDIR /synthea

# Rendre le script gradlew exécutable (au cas où)
RUN chmod +x ./gradlew

# Configurer les variables d'environnement pour Java/Gradle
ENV GRADLE_OPTS="-Xmx2g -Xms1g -XX:MaxMetaspaceSize=512m"
ENV JAVA_OPTS="-Xmx4g -Xms2g -XX:+UseG1GC"

# Build du projet Java en sautant les tests pour éviter les problèmes de mémoire
# Synthea est un générateur de données Java, les tests peuvent être lourds
RUN ./gradlew build -x test --no-daemon --max-workers=1

# Alternative avec tests (si vous avez assez de mémoire) :
# RUN ./gradlew build --no-daemon --max-workers=1 -Dorg.gradle.jvmargs="-Xmx4g -Xms2g"

# Vérifier que le JAR a été créé
RUN ls -la build/libs/

# Générer les patients synthétiques au lancement
CMD ["java", "-jar", "build/libs/synthea-with-dependencies.jar", "-p", "1000"]