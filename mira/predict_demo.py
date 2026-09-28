from vitamind.ml.classifier import MiraMLClassifier


def main():
    classifier = MiraMLClassifier()

    print("VitaMind Mira V5 ML demo")
    print("Type 'quit' to exit.")
    print()

    while True:
        text = input("TEXT: ").strip()

        if text.lower() in {"quit", "exit"}:
            break

        if not text:
            continue

        result = classifier.predict(text)

        print()
        print("ROUTING LABEL:", result.routing_label)
        print("PATHWAY:", result.pathway)
        print("CRISIS FLAG:", result.crisis_flag)
        print("TOP INTERNAL SCORE:", round(result.top_score_internal, 3))
        print("SECOND LABEL:", result.second_label)
        print("SECOND SCORE:", round(result.second_score_internal, 3))
        print("MARGIN:", round(result.margin_internal, 3))
        print()
        print("CLASS SCORES:")
        for label, score in result.class_scores_internal.items():
            print(f"  {label}: {score:.3f}")
        print()


if __name__ == "__main__":
    main()