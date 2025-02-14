import argparse
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

def main():
    parser = argparse.ArgumentParser(description="Pipeline de machine learning pour la prédiction de churn.")
    parser.add_argument('--prepare', action='store_true', help="Préparer les données.")
    parser.add_argument('--train', action='store_true', help="Entraîner le modèle.")
    parser.add_argument('--evaluate', action='store_true', help="Évaluer le modèle.")
    parser.add_argument('--save', action='store_true', help="Sauvegarder le modèle.")
    parser.add_argument('--load', action='store_true', help="Charger le modèle.")
    parser.add_argument('--data_path', type=str, help="Chemin vers le fichier de données.")
    parser.add_argument('--model_path', type=str, help="Chemin pour sauvegarder/charger le modèle.")
    
    args = parser.parse_args()
    
    if args.prepare:
        X_train, X_test, y_train, y_test = prepare_data(args.data_path)
        print("Données préparées avec succès.")
    
    if args.train:
        X_train, X_test, y_train, y_test = prepare_data(args.data_path)
        model = train_model(X_train, y_train)
        print("Modèle entraîné avec succès.")
    
    if args.evaluate:
        X_train, X_test, y_train, y_test = prepare_data(args.data_path)
        model = load_model(args.model_path)
        evaluate_model(model, X_test, y_test)
    
    if args.save:
        X_train, X_test, y_train, y_test = prepare_data(args.data_path)
        model = train_model(X_train, y_train)
        save_model(model, args.model_path)
        print(f"Modèle sauvegardé à {args.model_path}.")
    
    if args.load:
        model = load_model(args.model_path)
        print(f"Modèle chargé depuis {args.model_path}.")

if __name__ == "__main__":
    main()

