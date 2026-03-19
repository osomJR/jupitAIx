export const commonTranslations = {
  en: {
    back: "Back",
    language: "Language",
    english: "English",
    french: "Français",
    chooseFile: "Choose file",
    fileAccepted: "File accepted",
    formatPolicy: "Format policy",
    previewArea: "Preview area",
    outputFormat: "Output format",
    generating: "Generating...",
    converting: "Converting...",
    translating: "Translating...",
    explaining: "Explaining...",
    summarize: "Summarize",
    explain: "Explain",
    translate: "Translate",
    convert: "Convert",
  },
  fr: {
    back: "Retour",
    language: "Langue",
    english: "English",
    french: "Français",
    chooseFile: "Choisir un fichier",
    fileAccepted: "Fichier accepté",
    formatPolicy: "Règles de format",
    previewArea: "Zone d’aperçu",
    outputFormat: "Format de sortie",
    generating: "Génération...",
    converting: "Conversion...",
    translating: "Traduction...",
    explaining: "Explication...",
    summarize: "Résumer",
    explain: "Expliquer",
    translate: "Traduire",
    convert: "Convertir",
  },
};

export const actionCardTranslations = {
  en: {
    open: "Open",
    accountRequired: "Account required",
  },
  fr: {
    open: "Ouvrir",
    accountRequired: "Compte requis",
  },
};

export const homePageTranslations = {
  en: {
    badge: "Fast AI tools for everyday work",
    heroTitleStart: "Create, refine, and understand content",
    heroTitleHighlight: "in seconds",
    heroDescription:
      "Use powerful AI actions to convert, summarize, correct, translate, and explain content beautifully. Sign in to unlock even more advanced tools.",
    availableNowTitle: "Available now",
    availableNowDescription: "These features are ready for anonymous users.",
    unlockMoreEyebrow: "Unlock more",
    unlockMoreTitle: "Sign in or create an account to access advanced features",
    unlockMoreDescription:
      "Get access to premium AI capabilities like transcription and automatic question generation, designed for deeper workflows and more serious productivity.",
    signIn: "Sign In",
    signUp: "Sign Up",
    advancedFeaturesTitle: "Advanced features",
    advancedFeaturesDescription: "Available after sign in.",
    languageLabel: "Language",
    english: "English",
    french: "Français",
    enabledActions: [
      {
        key: "convert",
        name: "Convert",
        route: "/convert",
        description: "Transform files and content into the format you need.",
      },
      {
        key: "summarize",
        name: "Summarize",
        route: "/summarize",
        description: "Turn long content into sharp, useful highlights.",
      },
      {
        key: "grammar",
        name: "Grammar Correct",
        route: "/grammar",
        description: "Polish your writing with clean, confident corrections.",
      },
      {
        key: "translate",
        name: "Translate",
        route: "/translate",
        description: "Translate text naturally across multiple languages.",
      },
      {
        key: "explain",
        name: "Explain",
        route: "/explain",
        description: "Break down difficult ideas into simple explanations.",
      },
    ],
    lockedActions: [
      {
        key: "transcribe",
        name: "Transcribe",
        route: "/transcribe",
        description: "Convert audio and speech into accurate text.",
      },
      {
        key: "questions",
        name: "Generate Questions",
        route: "/questions",
        description: "Create smart questions from notes, text, or topics.",
      },
    ],
  },
  fr: {
    badge: "Des outils IA rapides pour le travail quotidien",
    heroTitleStart: "Créez, améliorez et comprenez le contenu",
    heroTitleHighlight: "en quelques secondes",
    heroDescription:
      "Utilisez de puissantes actions IA pour convertir, résumer, corriger, traduire et expliquer le contenu avec élégance. Connectez-vous pour débloquer encore plus d’outils avancés.",
    availableNowTitle: "Disponible maintenant",
    availableNowDescription:
      "Ces fonctionnalités sont prêtes pour les utilisateurs anonymes.",
    unlockMoreEyebrow: "Débloquez plus",
    unlockMoreTitle:
      "Connectez-vous ou créez un compte pour accéder aux fonctionnalités avancées",
    unlockMoreDescription:
      "Accédez à des capacités IA premium comme la transcription et la génération automatique de questions, conçues pour des flux de travail plus poussés et une productivité plus sérieuse.",
    signIn: "Se connecter",
    signUp: "S’inscrire",
    advancedFeaturesTitle: "Fonctionnalités avancées",
    advancedFeaturesDescription: "Disponibles après connexion.",
    languageLabel: "Langue",
    english: "English",
    french: "Français",
    enabledActions: [
      {
        key: "convert",
        name: "Convertir",
        route: "/convert",
        description:
          "Transformez vos fichiers et contenus dans le format souhaité.",
      },
      {
        key: "summarize",
        name: "Résumer",
        route: "/summarize",
        description:
          "Transformez un long contenu en points clés utiles et précis.",
      },
      {
        key: "grammar",
        name: "Corriger la grammaire",
        route: "/grammar",
        description:
          "Améliorez votre écriture avec des corrections claires et sûres.",
      },
      {
        key: "translate",
        name: "Traduire",
        route: "/translate",
        description: "Traduisez naturellement du texte dans plusieurs langues.",
      },
      {
        key: "explain",
        name: "Expliquer",
        route: "/explain",
        description: "Décomposez les idées difficiles en explications simples.",
      },
    ],
    lockedActions: [
      {
        key: "transcribe",
        name: "Transcrire",
        route: "/transcribe",
        description:
          "Convertissez l’audio et la parole en texte avec précision.",
      },
      {
        key: "questions",
        name: "Générer des questions",
        route: "/questions",
        description:
          "Créez des questions intelligentes à partir de notes, de texte ou de sujets.",
      },
    ],
  },
};
export const convertPageTranslations = {
  en: {
    badge: "Convert documents and images",
    title: "Convert files across the supported formats you need",
    description:
      "Upload a PDF, Word document, JPG, JPEG, or PNG. Only allowed conversion paths can be selected.",
    uploadTitle: "Upload a file to convert",
    allowedInputs: "Allowed inputs:",
    conversionOutput: "Conversion output",
    previewText:
      "Your conversion result will appear here after you choose a valid input file and an allowed target format.",

    unsupportedFileType:
      "Unsupported file type: {ext}. Only .pdf, .docx, .jpg, .jpeg, and .png are allowed.",
    fileTooLarge: "File is too large. Maximum allowed size is {maxSize} MB.",
    chooseFileToConvert: "Please choose a file to convert.",
    invalidConversion: "This conversion combination is not allowed.",
    conversionFailed: "Something went wrong while converting the file.",

    conversionCompleted: "Conversion completed.",
    inputFile: "Input file",
    inputExtension: "Input extension",
    outputExtension: "Output extension",
    conversionMatchesRules:
      "The selected conversion matches the allowed document conversion rules.",

    detectedType: "Detected type:",
    from: "From",
    convertTo: "Convert to",
    allowedOutputsFor: "Allowed outputs for",
    none: "none",
    conversionLabel: "Conversion:",

    allowedConversions: "Allowed conversions",
    strictConversionMatrix: "Strict conversion matrix",
    inputCoverage: "Input coverage",
    supportedUploadTypes: "Supported upload types",

    pdfWordTitle: "PDF ↔ Word",
    pdfWordDescription: ".pdf → .docx and .docx → .pdf",
    jpgWordPdfTitle: "JPG / JPEG → PDF or Word",
    jpgWordPdfDescription: ".jpg / .jpeg → .pdf or .docx",
    pngJpgTitle: "PNG → JPG / JPEG",
    pngJpgDescription: ".png → .jpg or .jpeg",

    pdfDocument: "PDF document",
    wordDocument: "Word document",
    jpgImage: "JPG image",
    pngImage: "PNG image",
    unknownFile: "Unknown file",
  },
  fr: {
    badge: "Convertir des documents et des images",
    title: "Convertissez vos fichiers selon les formats pris en charge",
    description:
      "Téléversez un PDF, un document Word, un JPG, un JPEG ou un PNG. Seuls les chemins de conversion autorisés peuvent être sélectionnés.",
    uploadTitle: "Téléversez un fichier à convertir",
    allowedInputs: "Entrées autorisées :",
    conversionOutput: "Résultat de la conversion",
    previewText:
      "Le résultat de la conversion apparaîtra ici après avoir choisi un fichier valide et un format cible autorisé.",

    unsupportedFileType:
      "Type de fichier non pris en charge : {ext}. Seuls les formats .pdf, .docx, .jpg, .jpeg et .png sont autorisés.",
    fileTooLarge:
      "Le fichier est trop volumineux. La taille maximale autorisée est de {maxSize} Mo.",
    chooseFileToConvert: "Veuillez choisir un fichier à convertir.",
    invalidConversion: "Cette combinaison de conversion n’est pas autorisée.",
    conversionFailed:
      "Une erreur s’est produite lors de la conversion du fichier.",

    conversionCompleted: "Conversion terminée.",
    inputFile: "Fichier d’entrée",
    inputExtension: "Extension d’entrée",
    outputExtension: "Extension de sortie",
    conversionMatchesRules:
      "La conversion sélectionnée respecte les règles de conversion autorisées.",

    detectedType: "Type détecté :",
    from: "De",
    convertTo: "Convertir vers",
    allowedOutputsFor: "Sorties autorisées pour",
    none: "aucune",
    conversionLabel: "Conversion :",

    allowedConversions: "Conversions autorisées",
    strictConversionMatrix: "Matrice de conversion stricte",
    inputCoverage: "Couverture des entrées",
    supportedUploadTypes: "Types de fichiers pris en charge",

    pdfWordTitle: "PDF ↔ Word",
    pdfWordDescription: ".pdf → .docx et .docx → .pdf",
    jpgWordPdfTitle: "JPG / JPEG → PDF ou Word",
    jpgWordPdfDescription: ".jpg / .jpeg → .pdf ou .docx",
    pngJpgTitle: "PNG → JPG / JPEG",
    pngJpgDescription: ".png → .jpg ou .jpeg",

    pdfDocument: "Document PDF",
    wordDocument: "Document Word",
    jpgImage: "Image JPG",
    pngImage: "Image PNG",
    unknownFile: "Fichier inconnu",
  },
};
