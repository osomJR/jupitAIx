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
    download: "Download output",
    transcribe: "Transcribe",
    transcribing: "Transcribing...",
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
    download: "Télécharger le fichier",
    transcribe: "Transcrire",
    transcribing: "Transcription...",
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
    loading: "Loading...",
    signedInAs: "Signed in as",
    logout: "Logout",
    advancedFeaturesTitle: "Advanced features",
    advancedFeaturesDescription: "Available after sign in.",
    unlockedSignedInTitle: "Advanced features unlocked",
    unlockedSignedInDescription:
      "You can now access transcription, question generation, and other protected tools.",
    advancedFeaturesSignedInDescription: "Available now.",
    checkingAccountStatus: "Checking account status...",
    languageLabel: "Language",
    english: "English",
    french: "Français",
    settings: "Settings",
    appearance: "Appearance",
    light: "Light",
    dark: "Dark",
    systemDefault: "System Default",
    enabledActions: [
      {
        key: "convert",
        name: "File Conversion",
        route: "/convert",
        description: "Convert files and documents to the formats you need",
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
        name: "Transcribe audio and video",
        route: "/transcribe",
        description: "Convert speech to accurate text.",
      },
      {
        key: "questions",
        name: "Generate Questions",
        route: "/questions",
        description: "Create smart questions from notes, text, or topics.",
      },
      {
        key: "redact",
        name: "Redact",
        route: "/redact",
        description: "XYZ",
      },
      {
        key: "mask",
        name: "Data Mask",
        route: "/mask",
        description: "XYZ",
      },
      {
        key: "compliance",
        name: "Compliance",
        route: "/compliance",
        description: "XYZ",
      },
      {
        key: "extraction",
        name: "Structured Extraction",
        route: "/extraction",
        description: "XYZ",
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
    loading: "Chargement...",
    signedInAs: "Connecté en tant que",
    logout: "Se déconnecter",
    advancedFeaturesTitle: "Fonctionnalités avancées",
    advancedFeaturesDescription: "Disponibles après connexion.",
    unlockedSignedInTitle: "Fonctionnalités avancées débloquées",
    unlockedSignedInDescription:
      "Vous pouvez maintenant accéder à la transcription, à la génération de questions et à d’autres outils protégés.",
    advancedFeaturesSignedInDescription: "Disponible maintenant.",
    checkingAccountStatus: "Vérification du statut du compte...",
    languageLabel: "Langue",
    english: "English",
    french: "Français",
    settings: "Paramètres",
    appearance: "Apparence",
    light: "Clair",
    dark: "Sombre",
    systemDefault: "Par défaut du système",
    enabledActions: [
      {
        key: "convert",
        name: "Conversion de fichier",
        route: "/convert",
        description:
          "Convertissez vos fichiers et documents dans les formats dont vouz avez besoin",
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
        name: "Transcrire des fichiers audio et vidéo",
        route: "/transcribe",
        description: "Convertir la parole en texte avec précision.",
      },
      {
        key: "questions",
        name: "Générer des questions",
        route: "/questions",
        description:
          "Créez des questions intelligentes à partir de notes, de texte ou de sujets.",
      },
      {
        key: "redact",
        name: "Caviarder",
        route: "/redact",
        description: "XYZ",
      },
      {
        key: "mask",
        name: "Masquage des données",
        route: "/mask",
        description: "XYZ",
      },
      {
        key: "compliance",
        name: "Conformité",
        route: "/compliance",
        description: "XYZ",
      },
      {
        key: "extraction",
        name: "Extraction structurée",
        route: "/extraction",
        description: "XYZ",
      },
    ],
  },
};
export const convertPageTranslations = {
  en: {
    badge: "Convert documents, files and images",
    title: "Convert files across several formats",
    description: "Upload PDF, Word document, JPG, JPEG, or PNG.",
    uploadTitle: "Upload file or document",
    conversionOutput: "Conversion result",
    previewText: "Download appears here after file conversion.",

    unsupportedFileType:
      "Unsupported file type: {ext}. Only .pdf, .docx, .jpg, .jpeg, and .png are allowed.",
    fileTooLarge: "File is too large. Maximum allowed size is {maxSize} MB.",
    chooseFileToConvert: "Please choose a file to convert.",
    invalidConversion: "This conversion combination is not allowed.",
    conversionFailed: "Something went wrong while converting the file.",
    missingDownloadUrl:
      "Conversion finished, but the backend did not return a download URL.",

    conversionCompleted: "Conversion completed.",
    inputFile: "Input file",
    inputExtension: "Input extension",
    outputExtension: "Output extension",
    downloadReady: "Download ready",
    convertedFile: "Converted file",
    outputReadyText: "Your converted file is ready for download.",
    detectedType: "Detected type:",
    from: "From",
    convertTo: "Convert to",
    allowedOutputsFor: "Allowed outputs for",
    none: "none",
    conversionLabel: "Conversion:",

    pdfDocument: "PDF document",
    wordDocument: "Word document",
    jpgImage: "JPG image",
    pngImage: "PNG image",
    unknownFile: "Unknown file",
  },
  fr: {
    badge: "Convertir des documents, fichiers et images",
    title: "Convertir des fichiers dans plusieurs formats",
    description:
      "Téléversez un PDF, un document Word, un JPG, un JPEG ou un PNG.",
    uploadTitle: "Téléverser un fichier ou un document",
    conversionOutput: "Résultat de la conversion",
    previewText:
      "Le téléchargement apparaîtra ici après la conversion du fichier.",

    unsupportedFileType:
      "Type de fichier non pris en charge : {ext}. Seuls les formats .pdf, .docx, .jpg, .jpeg et .png sont autorisés.",
    fileTooLarge:
      "Le fichier est trop volumineux. La taille maximale autorisée est de {maxSize} Mo.",
    chooseFileToConvert: "Veuillez choisir un fichier à convertir.",
    invalidConversion: "Cette combinaison de conversion n’est pas autorisée.",
    conversionFailed:
      "Une erreur s’est produite lors de la conversion du fichier.",
    missingDownloadUrl:
      "La conversion est terminée, mais le backend n’a pas renvoyé d’URL de téléchargement.",

    conversionCompleted: "Conversion terminée.",
    inputFile: "Fichier d’entrée",
    inputExtension: "Extension d’entrée",
    outputExtension: "Extension de sortie",
    downloadReady: "Téléchargement prêt",
    convertedFile: "Fichier converti",
    outputReadyText: "Votre fichier converti est prêt à être téléchargé.",
    detectedType: "Type détecté :",
    from: "De",
    convertTo: "Convertir vers",
    allowedOutputsFor: "Sorties autorisées pour",
    none: "aucune",
    conversionLabel: "Conversion :",

    pdfDocument: "Document PDF",
    wordDocument: "Document Word",
    jpgImage: "Image JPG",
    pngImage: "Image PNG",
    unknownFile: "Fichier inconnu",
  },
};
export const explainPageTranslations = {
  en: {
    badge: "Explain content clearly",
    title: "Break down difficult content into simple explanations",
    description:
      "Upload a PDF or Word document, or paste inline text. Unsupported files like PNG are rejected automatically, and the output extension always matches the input extension.",

    fileMode: "Upload file",
    textMode: "Inline text",

    uploadTitle: "Upload content to explain",
    allowedFileInputs:
      "Allowed: .pdf and .docx. Rejected automatically: .png, .jpg, and unsupported formats.",
    outputExtensionWillBe: "Output extension will be",

    pasteTextLabel: "Paste text to explain",
    pasteTextPlaceholder: "Paste or type your text here...",
    inlineTextTreatedAs:
      "Inline text is treated as .txt, so the output extension will also be .txt.",

    unsupportedFileType:
      "Unsupported file type: {ext}. Only .pdf and .docx uploads are allowed. PNG and other image formats are rejected.",
    fileTooLarge: "File is too large. Maximum allowed size is {maxSize} MB.",
    explanationFailed: "Something went wrong while generating the explanation.",

    generatingExplanation: "Generating explanation...",
    outputFormatLabel: "Output format:",

    policyTitle: "Format policy",
    policySubtitle: "Strict input and output matching",
    allowedUploadsLabel: "Allowed uploads:",
    inlineInputLabel: "Inline input:",
    rejectedAutomaticallyLabel: "Rejected automatically:",
    outputRuleLabel: "Output rule:",
    inlineInputValue: "treated as .txt",
    rejectedAutomaticallyValue: ".png and all unsupported file types",
    outputRuleValue: "output extension must always equal input extension",

    explanationOutputTitle: "Explanation output",
    previewEmpty:
      "Your generated explanation will appear here. The output extension always mirrors the original input extension.",
    outputExtensionLabel: "Output extension:",

    inlineExplanationIntro: "Explanation generated from inline text.",
    fileExplanationIntro: "Explanation generated from {filename}.",
    rewrittenPreviewText:
      "This content has been rewritten into a simpler explanation while keeping the same output format rule.",
    previewLabel: "Preview:",
    inputExtensionLabel: "Input extension:",
    outputExtensionResultLabel: "Output extension:",
    preservedExtensionMessage:
      "The explanation output preserves the same extension as the original uploaded file.",
  },
  fr: {
    badge: "Expliquer clairement le contenu",
    title: "Décomposez les contenus difficiles en explications simples",
    description:
      "Téléversez un PDF ou un document Word, ou collez du texte inline. Les fichiers non pris en charge comme PNG sont rejetés automatiquement, et l’extension de sortie correspond toujours à l’extension d’entrée.",

    fileMode: "Téléverser un fichier",
    textMode: "Texte inline",

    uploadTitle: "Téléverser un contenu à expliquer",
    allowedFileInputs:
      "Autorisés : .pdf et .docx. Rejetés automatiquement : .png, .jpg et les formats non pris en charge.",
    outputExtensionWillBe: "L’extension de sortie sera",

    pasteTextLabel: "Coller le texte à expliquer",
    pasteTextPlaceholder: "Collez ou saisissez votre texte ici...",
    inlineTextTreatedAs:
      "Le texte inline est traité comme .txt, donc l’extension de sortie sera également .txt.",

    unsupportedFileType:
      "Type de fichier non pris en charge : {ext}. Seuls les fichiers .pdf et .docx sont autorisés. Les formats PNG et autres images sont rejetés.",
    fileTooLarge:
      "Le fichier est trop volumineux. La taille maximale autorisée est de {maxSize} Mo.",
    explanationFailed:
      "Une erreur s’est produite lors de la génération de l’explication.",

    generatingExplanation: "Génération de l’explication...",
    outputFormatLabel: "Format de sortie :",

    policyTitle: "Règles de format",
    policySubtitle: "Correspondance stricte entre entrée et sortie",
    allowedUploadsLabel: "Téléversements autorisés :",
    inlineInputLabel: "Entrée inline :",
    rejectedAutomaticallyLabel: "Rejetés automatiquement :",
    outputRuleLabel: "Règle de sortie :",
    inlineInputValue: "traité comme .txt",
    rejectedAutomaticallyValue:
      ".png et tous les types de fichiers non pris en charge",
    outputRuleValue:
      "l’extension de sortie doit toujours être identique à l’extension d’entrée",

    explanationOutputTitle: "Résultat de l’explication",
    previewEmpty:
      "Votre explication générée apparaîtra ici. L’extension de sortie reflète toujours l’extension d’entrée d’origine.",
    outputExtensionLabel: "Extension de sortie :",

    inlineExplanationIntro: "Explication générée à partir du texte inline.",
    fileExplanationIntro: "Explication générée à partir de {filename}.",
    rewrittenPreviewText:
      "Ce contenu a été reformulé en une explication plus simple tout en conservant la même règle de format de sortie.",
    previewLabel: "Aperçu :",
    inputExtensionLabel: "Extension d’entrée :",
    outputExtensionResultLabel: "Extension de sortie :",
    preservedExtensionMessage:
      "Le résultat de l’explication conserve la même extension que le fichier téléversé d’origine.",
  },
};
export const summarizePageTranslations = {
  en: {
    badge: "Summarize content",
    title: "Summarize documents or text with strict format rules",
    description:
      "Upload a PDF or Word document, or paste inline text. Unsupported files like PNG are automatically rejected, and the output extension always matches the input extension.",

    fileMode: "Upload file",
    textMode: "Inline text",

    uploadTitle: "Upload a supported document",
    allowedFileInputs:
      "Allowed: .pdf and .docx. Rejected automatically: .png, .jpg, and all unsupported formats.",
    outputExtensionWillBe: "Output extension will be",

    pasteTextLabel: "Paste text to summarize",
    pasteTextPlaceholder: "Paste or type your text here...",
    inlineTextTreatedAs:
      "Inline text is treated as .txt, so the output extension will also be .txt.",

    unsupportedFileType:
      "Unsupported file type: {ext}. Only .pdf and .docx uploads are allowed. PNG and other image formats are rejected.",
    fileTooLarge: "File is too large. Maximum allowed size is {maxSize} MB.",
    summaryFailed: "Something went wrong while generating the summary.",

    generatingSummary: "Generating summary...",

    policySubtitle: "Strict input and output matching",
    allowedUploadsLabel: "Allowed uploads:",
    inlineInputLabel: "Inline input:",
    rejectedAutomaticallyLabel: "Rejected automatically:",
    outputRuleLabel: "Output rule:",
    inlineInputValue: "treated as .txt",
    rejectedAutomaticallyValue: ".png and all unsupported file types",
    outputRuleValue: "output extension must always equal input extension",

    summaryOutputTitle: "Summary output",
    previewEmpty:
      "Your generated summary will appear here. The output extension will always mirror the original input extension.",
    outputExtensionLabel: "Output extension:",

    inlineSummaryIntro: "Summary generated from inline text.",
    fileSummaryIntro: "Summary generated from {filename}.",
    translatedPreviewLabel: "Preview:",
    inputExtensionLabel: "Input extension:",
    outputExtensionResultLabel: "Output extension:",
    preservedExtensionMessage:
      "The output format remains the same as the uploaded file format.",
  },
  fr: {
    badge: "Résumer le contenu",
    title:
      "Résumez des documents ou du texte avec des règles de format strictes",
    description:
      "Téléversez un PDF ou un document Word, ou collez du texte inline. Les fichiers non pris en charge comme PNG sont automatiquement rejetés, et l’extension de sortie correspond toujours à l’extension d’entrée.",

    fileMode: "Téléverser un fichier",
    textMode: "Texte inline",

    uploadTitle: "Téléverser un document pris en charge",
    allowedFileInputs:
      "Autorisés : .pdf et .docx. Rejetés automatiquement : .png, .jpg et tous les formats non pris en charge.",
    outputExtensionWillBe: "L’extension de sortie sera",

    pasteTextLabel: "Coller le texte à résumer",
    pasteTextPlaceholder: "Collez ou saisissez votre texte ici...",
    inlineTextTreatedAs:
      "Le texte inline est traité comme .txt, donc l’extension de sortie sera également .txt.",

    unsupportedFileType:
      "Type de fichier non pris en charge : {ext}. Seuls les fichiers .pdf et .docx sont autorisés. Les formats PNG et autres images sont rejetés.",
    fileTooLarge:
      "Le fichier est trop volumineux. La taille maximale autorisée est de {maxSize} Mo.",
    summaryFailed: "Une erreur s’est produite lors de la génération du résumé.",

    generatingSummary: "Génération du résumé...",

    policySubtitle: "Correspondance stricte entre entrée et sortie",
    allowedUploadsLabel: "Téléversements autorisés :",
    inlineInputLabel: "Entrée inline :",
    rejectedAutomaticallyLabel: "Rejetés automatiquement :",
    outputRuleLabel: "Règle de sortie :",
    inlineInputValue: "traité comme .txt",
    rejectedAutomaticallyValue:
      ".png et tous les types de fichiers non pris en charge",
    outputRuleValue:
      "l’extension de sortie doit toujours être identique à l’extension d’entrée",

    summaryOutputTitle: "Résultat du résumé",
    previewEmpty:
      "Votre résumé généré apparaîtra ici. L’extension de sortie reflétera toujours l’extension d’entrée d’origine.",
    outputExtensionLabel: "Extension de sortie :",

    inlineSummaryIntro: "Résumé généré à partir du texte inline.",
    fileSummaryIntro: "Résumé généré à partir de {filename}.",
    translatedPreviewLabel: "Aperçu :",
    inputExtensionLabel: "Extension d’entrée :",
    outputExtensionResultLabel: "Extension de sortie :",
    preservedExtensionMessage:
      "Le format de sortie reste identique à celui du fichier téléversé.",
  },
};

export const translatePageTranslations = {
  en: {
    badge: "Translate content naturally",
    title: "Translate documents or text while preserving format rules",
    description:
      "Upload a PDF or Word document, or paste inline text. Unsupported files like PNG are rejected automatically, and the output extension always matches the input extension.",

    fileMode: "Upload file",
    textMode: "Inline text",

    targetLanguageLabel: "Translate to",
    targetLanguagePlaceholder:
      "Enter any target language, e.g. French, Yoruba, Japanese",
    targetLanguageHelp:
      "Enter any target language supported by the model instead of choosing from a limited dropdown.",

    uploadTitle: "Upload content to translate",
    allowedFileInputs:
      "Allowed: .pdf and .docx. Rejected automatically: .png, .jpg, and unsupported formats.",
    outputExtensionWillBe: "Output extension will be",

    pasteTextLabel: "Paste text to translate",
    pasteTextPlaceholder: "Paste or type your text here...",
    inlineTextTreatedAs:
      "Inline text is treated as .txt, so the output extension will also be .txt.",

    unsupportedFileType:
      "Unsupported file type: {ext}. Only .pdf and .docx uploads are allowed. PNG and other image formats are rejected.",
    fileTooLarge: "File is too large. Maximum allowed size is {maxSize} MB.",
    targetLanguageRequired: "Please enter a target language.",
    translationFailed: "Something went wrong while generating the translation.",

    generatingTranslation: "Generating translation...",

    policySubtitle: "Strict input and output matching",
    allowedUploadsLabel: "Allowed uploads:",
    inlineInputLabel: "Inline input:",
    rejectedAutomaticallyLabel: "Rejected automatically:",
    outputRuleLabel: "Output rule:",
    inlineInputValue: "treated as .txt",
    rejectedAutomaticallyValue: ".png and all unsupported file types",
    outputRuleValue: "output extension must always equal input extension",

    translationOutputTitle: "Translation output",
    previewEmpty:
      "Your generated translation will appear here. The output extension always mirrors the original input extension.",
    outputExtensionLabel: "Output extension:",

    inlineTranslationIntro: "Translation generated from inline text.",
    fileTranslationIntro: "Translation generated from {filename}.",
    targetLanguageResultLabel: "Target language:",
    translatedPreviewLabel: "Translated preview:",
    inputExtensionLabel: "Input extension:",
    outputExtensionResultLabel: "Output extension:",
    preservedExtensionMessage:
      "The translated output keeps the same extension as the uploaded input.",

    languageSuggestions: [
      "English",
      "French",
      "Spanish",
      "German",
      "Portuguese",
      "Brazilian Portuguese",
      "Arabic",
      "Chinese",
      "Simplified Chinese",
      "Traditional Chinese",
      "Japanese",
      "Korean",
      "Hindi",
      "Yoruba",
      "Hausa",
      "Igbo",
      "Swahili",
      "Turkish",
      "Russian",
      "Italian",
      "Dutch",
    ],
  },
  fr: {
    badge: "Traduire le contenu naturellement",
    title:
      "Traduisez des documents ou du texte tout en préservant les règles de format",
    description:
      "Téléversez un PDF ou un document Word, ou collez du texte inline. Les fichiers non pris en charge comme PNG sont automatiquement rejetés, et l’extension de sortie correspond toujours à l’extension d’entrée.",

    fileMode: "Téléverser un fichier",
    textMode: "Texte inline",

    targetLanguageLabel: "Traduire vers",
    targetLanguagePlaceholder:
      "Saisissez une langue cible, par ex. français, yoruba, japonais",
    targetLanguageHelp:
      "Saisissez toute langue cible prise en charge par le modèle au lieu de choisir dans une liste limitée.",

    uploadTitle: "Téléverser un contenu à traduire",
    allowedFileInputs:
      "Autorisés : .pdf et .docx. Rejetés automatiquement : .png, .jpg et les formats non pris en charge.",
    outputExtensionWillBe: "L’extension de sortie sera",

    pasteTextLabel: "Coller le texte à traduire",
    pasteTextPlaceholder: "Collez ou saisissez votre texte ici...",
    inlineTextTreatedAs:
      "Le texte inline est traité comme .txt, donc l’extension de sortie sera également .txt.",

    unsupportedFileType:
      "Type de fichier non pris en charge : {ext}. Seuls les fichiers .pdf et .docx sont autorisés. Les formats PNG et autres images sont rejetés.",
    fileTooLarge:
      "Le fichier est trop volumineux. La taille maximale autorisée est de {maxSize} Mo.",
    targetLanguageRequired: "Veuillez saisir une langue cible.",
    translationFailed:
      "Une erreur s’est produite lors de la génération de la traduction.",

    generatingTranslation: "Génération de la traduction...",

    policySubtitle: "Correspondance stricte entre entrée et sortie",
    allowedUploadsLabel: "Téléversements autorisés :",
    inlineInputLabel: "Entrée inline :",
    rejectedAutomaticallyLabel: "Rejetés automatiquement :",
    outputRuleLabel: "Règle de sortie :",
    inlineInputValue: "traité comme .txt",
    rejectedAutomaticallyValue:
      ".png et tous les types de fichiers non pris en charge",
    outputRuleValue:
      "l’extension de sortie doit toujours être identique à l’extension d’entrée",

    translationOutputTitle: "Résultat de la traduction",
    previewEmpty:
      "Votre traduction générée apparaîtra ici. L’extension de sortie reflète toujours l’extension d’entrée d’origine.",
    outputExtensionLabel: "Extension de sortie :",

    inlineTranslationIntro: "Traduction générée à partir du texte inline.",
    fileTranslationIntro: "Traduction générée à partir de {filename}.",
    targetLanguageResultLabel: "Langue cible :",
    translatedPreviewLabel: "Aperçu traduit :",
    inputExtensionLabel: "Extension d’entrée :",
    outputExtensionResultLabel: "Extension de sortie :",
    preservedExtensionMessage:
      "Le résultat traduit conserve la même extension que l’entrée téléversée.",

    languageSuggestions: [
      "English",
      "French",
      "Spanish",
      "German",
      "Portuguese",
      "Brazilian Portuguese",
      "Arabic",
      "Chinese",
      "Simplified Chinese",
      "Traditional Chinese",
      "Japanese",
      "Korean",
      "Hindi",
      "Yoruba",
      "Hausa",
      "Igbo",
      "Swahili",
      "Turkish",
      "Russian",
      "Italian",
      "Dutch",
    ],
  },
};
export const transcribePageTranslations = {
  en: {
    badge: "Transcription",
    title: "Transcribe audio and video",
    description: "Upload audio (.mp3) or video (.mp4, .mkv, .mov).",
    uploadTitle: "Upload audio or video",
    allowedFileInputs: "Allowed inputs: .mp3, .mp4, .mkv, .mov",
    unsupportedFileType: "Unsupported file type: {ext}",
    fileTooLarge:
      "File is too large. Maximum size for this media type is {maxSize} MB.",
    mediaTooLong:
      "Media is too long. Maximum duration for this media type is {maxDuration}.",
    couldNotReadDuration:
      "Could not read media duration. Please try another file.",
    chooseFileToTranscribe: "Please choose an audio or video file.",
    transcriptionFailed: "Transcription request failed.",
    validatingMedia: "Checking media...",
    transcriptOutput: "Transcript output",
    previewText: "Your transcript will appear here after processing.",

    transcriptOptionsTitle: "Transcription options",
    transcriptOptionsSubtitle: "Choose how the transcript should be processed.",
    preserveFillerWordsLabel: "Preserve filler words",
    preserveFillerWordsHelp: "Keep words like “um”, “uh”, and similar fillers.",
    removeBackgroundNoiseLabel: "Remove background noise",
    removeBackgroundNoiseHelp:
      "Apply optional minimal background-noise cleanup.",
    diarizeSpeakersLabel: "Separate speakers",
    diarizeSpeakersHelp:
      "Separate speakers only when they are acoustically detectable.",

    transcriptReady: "Transcript ready",
    transcriptReadyText:
      "The spoken content has been converted into written text.",
    transcriptMetaLabel: "Transcript format",
    transcriptMetaValue: ".txt inline text only",
    detectedTypeLabel: "Detected media type",
    durationLabel: "Duration",
    audioType: "Audio",
    videoType: "Video",
    unknownType: "Unknown",
  },
  fr: {
    badge: "Transcription",
    title: "Transcrire l’audio et la vidéo",
    description:
      "Téléversez un fichier audio (.mp3) ou vidéo (.mp4, .mkv, .mov).",
    uploadTitle: "Téléverser un fichier audio ou vidéo",
    allowedFileInputs: "Entrées autorisées : .mp3, .mp4, .mkv, .mov",
    unsupportedFileType: "Type de fichier non pris en charge : {ext}",
    fileTooLarge:
      "Le fichier est trop volumineux. La taille maximale pour ce type de média est de {maxSize} Mo.",
    mediaTooLong:
      "Le média est trop long. La durée maximale pour ce type de média est de {maxDuration}.",
    couldNotReadDuration:
      "Impossible de lire la durée du média. Veuillez essayer un autre fichier.",
    chooseFileToTranscribe: "Veuillez choisir un fichier audio ou vidéo.",
    transcriptionFailed: "La requête de transcription a échoué.",
    validatingMedia: "Vérification du média...",
    transcriptOutput: "Résultat de la transcription",
    previewText: "Votre transcription apparaîtra ici après le traitement.",

    transcriptOptionsTitle: "Options de transcription",
    transcriptOptionsSubtitle:
      "Choisissez comment la transcription doit être traitée.",
    preserveFillerWordsLabel: "Préserver les mots de remplissage",
    preserveFillerWordsHelp:
      "Conserver les mots comme « euh », « hum » et équivalents.",
    removeBackgroundNoiseLabel: "Réduire le bruit de fond",
    removeBackgroundNoiseHelp:
      "Appliquer un nettoyage minimal et optionnel du bruit de fond.",
    diarizeSpeakersLabel: "Séparer les intervenants",
    diarizeSpeakersHelp:
      "Séparer les intervenants uniquement lorsqu’ils sont détectables acoustiquement.",

    transcriptReady: "Transcription prête",
    transcriptReadyText: "Le contenu parlé a été converti en texte écrit.",
    transcriptMetaLabel: "Format de transcription",
    transcriptMetaValue: ".txt texte inline uniquement",
    detectedTypeLabel: "Type de média détecté",
    durationLabel: "Durée",
    audioType: "Audio",
    videoType: "Vidéo",
    unknownType: "Inconnu",
  },
};
export const redactPageTranslations = {
  en: {
    badge: "Privacy-first black-box redaction",
    title: "Redact sensitive content while preserving the document structure",
    description:
      "Upload a PDF, Word document, or image. The first pass generates a provisional redacted file for review. Then you can deselect any grouped quote you do not want redacted before generating the final file.",
    uploadTitle: "Upload a document to redact",
    allowedFileInputs: "Allowed: .pdf, .docx, .jpg, .jpeg, .png.",
    outputExtensionWillBe: "Output extension will be",
    unsupportedFileType:
      "Unsupported file type: {ext}. Only .pdf, .docx, .jpg, .jpeg, and .png are allowed.",
    fileTooLarge: "File is too large. Maximum allowed size is {maxSize} MB.",
    chooseFileToRedact: "Please choose a file to redact.",
    redactionFailed: "Something went wrong while processing redaction.",
    redactAction: "Redact document",
    generating: "Redacting...",
    reviewing: "Processing review...",
    finalizing: "Finalizing...",
    processAndReview: "Process and review",
    finalizeAction: "Generate final file",
    resultTitle: "Redaction output",
    previewEmpty:
      "Your provisional or final redacted file will appear here together with grouped review items.",
    policySubtitle: "Strict privacy processing rules",
    allowedUploadsLabel: "Allowed uploads:",
    outputRuleLabel: "Output rule:",
    outputRuleValue: "output extension must always equal input extension",
    docTypeLabel: "Document type",
    sensitiveTargetsLabel: "Sensitive data to redact",
    exclusionsLabel: "Manual review exclusions",
    exclusionsPlaceholder:
      "Optional: enter words or phrases to always leave visible, one per line or comma-separated.",
    selectedTargetsLabel: "Selected targets:",
    fileAcceptedLabel: "File accepted",
    downloadReady: "Download ready",
    outputReadyText: "Your final redacted file is ready to download.",
    missingDownloadUrl:
      "Processing finished, but the backend did not return a download URL.",
    processedFile: "Processed file",
    inputFile: "Input file",
    inputExtension: "Input extension",
    outputExtension: "Output extension",
    documentTypeResult: "Document type",
    exclusionsCount: "Review exclusions count",
    rulesApplied: "Redaction was generated from the reviewed selection set.",
    fileTypeLabel: "Detected type:",
    selectAll: "Select all",
    clearAll: "Clear all",

    provisionalReady: "Provisional redacted file ready",
    finalReady: "Final redacted file ready",
    reviewTitle: "Review grouped redaction items",
    reviewHint:
      "Checked items stay redacted everywhere they appear. Unchecked items stay visible everywhere they appear.",
    reviewItemsLabel: "Grouped review items",
    processedPreviewTitle: "Processed document preview",
    docxPreviewNotice:
      "DOCX preview is shown using a generated PDF preview for review. Final download remains DOCX.",
    approveAll: "Approve all",
    clearApproved: "Clear all",
    approvedCountLabel: "Approved items",
    deselectedCountLabel: "Deselected items",
    occurrencesLabel: "Occurrences",
    noCandidates:
      "No grouped sensitive items were detected for the current settings.",
  },
  fr: {
    badge: "Caviardage boîte noire orienté confidentialité",
    title:
      "Caviardez le contenu sensible tout en préservant la structure du document",
    description:
      "Téléversez un PDF, un document Word ou une image. Le premier passage génère un fichier caviardé provisoire pour révision. Ensuite, vous pouvez désélectionner tout groupe de texte que vous ne souhaitez pas caviarder avant de générer le fichier final.",
    uploadTitle: "Téléverser un document à caviarder",
    allowedFileInputs: "Autorisés : .pdf, .docx, .jpg, .jpeg, .png.",
    outputExtensionWillBe: "L’extension de sortie sera",
    unsupportedFileType:
      "Type de fichier non pris en charge : {ext}. Seuls .pdf, .docx, .jpg, .jpeg et .png sont autorisés.",
    fileTooLarge:
      "Le fichier est trop volumineux. La taille maximale autorisée est de {maxSize} MB.",
    chooseFileToRedact: "Veuillez choisir un fichier à caviarder.",
    redactionFailed:
      "Une erreur s’est produite pendant le traitement du caviardage.",
    redactAction: "Caviarder le document",
    generating: "Caviardage...",
    reviewing: "Préparation de la révision...",
    finalizing: "Finalisation...",
    processAndReview: "Traiter et réviser",
    finalizeAction: "Générer le fichier final",
    resultTitle: "Sortie du caviardage",
    previewEmpty:
      "Votre fichier caviardé provisoire ou final apparaîtra ici avec les éléments groupés à réviser.",
    policySubtitle: "Règles strictes de traitement confidentiel",
    allowedUploadsLabel: "Téléversements autorisés :",
    outputRuleLabel: "Règle de sortie :",
    outputRuleValue:
      "l’extension de sortie doit toujours être identique à l’extension d’entrée",
    docTypeLabel: "Type de document",
    sensitiveTargetsLabel: "Données sensibles à caviarder",
    exclusionsLabel: "Exclusions manuelles de révision",
    exclusionsPlaceholder:
      "Optionnel : saisissez les mots ou expressions à toujours laisser visibles, une par ligne ou séparés par des virgules.",
    selectedTargetsLabel: "Cibles sélectionnées :",
    fileAcceptedLabel: "Fichier accepté",
    downloadReady: "Téléchargement prêt",
    outputReadyText: "Votre fichier caviardé final est prêt à être téléchargé.",
    missingDownloadUrl:
      "Le traitement est terminé, mais le backend n’a pas renvoyé d’URL de téléchargement.",
    processedFile: "Fichier traité",
    inputFile: "Fichier d’entrée",
    inputExtension: "Extension d’entrée",
    outputExtension: "Extension de sortie",
    documentTypeResult: "Type de document",
    exclusionsCount: "Nombre d’exclusions",
    rulesApplied:
      "Le caviardage a été généré à partir de l’ensemble sélectionné après révision.",
    fileTypeLabel: "Type détecté :",
    selectAll: "Tout sélectionner",
    clearAll: "Tout effacer",

    provisionalReady: "Fichier caviardé provisoire prêt",
    finalReady: "Fichier caviardé final prêt",
    reviewTitle: "Réviser les éléments groupés à caviarder",
    reviewHint:
      "Les éléments cochés restent caviardés partout où ils apparaissent. Les éléments décochés restent visibles partout où ils apparaissent.",
    reviewItemsLabel: "Éléments groupés à réviser",
    processedPreviewTitle: "Aperçu du document traité",
    docxPreviewNotice:
      "L’aperçu DOCX est affiché à l’aide d’un aperçu PDF généré pour la révision. Le téléchargement final reste en DOCX.",
    approveAll: "Tout approuver",
    clearApproved: "Tout effacer",
    approvedCountLabel: "Éléments approuvés",
    deselectedCountLabel: "Éléments désélectionnés",
    occurrencesLabel: "Occurrences",
    noCandidates:
      "Aucun élément sensible groupé n’a été détecté pour les paramètres actuels.",
  },
};

export const dataMaskPageTranslations = {
  en: {
    badge: "Privacy-first data masking",
    title: "Mask sensitive content while preserving the document structure",
    description:
      "Upload a PDF, Word document, or image. The first pass generates a provisional masked file for review. Then you can deselect any grouped quote you do not want masked before generating the final file.",
    uploadTitle: "Upload a document to mask",
    allowedFileInputs: "Allowed: .pdf, .docx, .jpg, .jpeg, .png.",
    outputExtensionWillBe: "Output extension will be",
    unsupportedFileType:
      "Unsupported file type: {ext}. Only .pdf, .docx, .jpg, .jpeg, and .png are allowed.",
    fileTooLarge: "File is too large. Maximum allowed size is {maxSize} MB.",
    chooseFileToMask: "Please choose a file to mask.",
    maskingFailed: "Something went wrong while processing data masking.",
    maskAction: "Mask document",
    generating: "Masking...",
    reviewing: "Processing review...",
    finalizing: "Finalizing...",
    processAndReview: "Process and review",
    finalizeAction: "Generate final file",
    resultTitle: "Data masking output",
    previewEmpty:
      "Your provisional or final masked file will appear here together with grouped review items.",
    policySubtitle: "Strict privacy processing rules",
    allowedUploadsLabel: "Allowed uploads:",
    outputRuleLabel: "Output rule:",
    outputRuleValue: "output extension must always equal input extension",
    docTypeLabel: "Document type",
    sensitiveTargetsLabel: "Sensitive data to mask",
    exclusionsLabel: "Manual review exclusions",
    exclusionsPlaceholder:
      "Optional: enter words or phrases to always leave visible, one per line or comma-separated.",
    selectedTargetsLabel: "Selected targets:",
    fileAcceptedLabel: "File accepted",
    downloadReady: "Download ready",
    outputReadyText: "Your final masked file is ready to download.",
    missingDownloadUrl:
      "Processing finished, but the backend did not return a download URL.",
    processedFile: "Processed file",
    inputFile: "Input file",
    inputExtension: "Input extension",
    outputExtension: "Output extension",
    documentTypeResult: "Document type",
    exclusionsCount: "Review exclusions count",
    rulesApplied: "Masking was generated from the reviewed selection set.",
    fileTypeLabel: "Detected type:",
    selectAll: "Select all",
    clearAll: "Clear all",

    provisionalReady: "Provisional masked file ready",
    finalReady: "Final masked file ready",
    reviewTitle: "Review grouped masking items",
    reviewHint:
      "Checked items stay masked everywhere they appear. Unchecked items stay visible everywhere they appear.",
    reviewItemsLabel: "Grouped review items",
    processedPreviewTitle: "Processed document preview",
    docxPreviewNotice:
      "DOCX preview is shown using a generated PDF preview for review. Final download remains DOCX.",
    approveAll: "Approve all",
    clearApproved: "Clear all",
    approvedCountLabel: "Approved items",
    deselectedCountLabel: "Deselected items",
    occurrencesLabel: "Occurrences",
    noCandidates:
      "No grouped sensitive items were detected for the current settings.",
  },
  fr: {
    badge: "Masquage de données orienté confidentialité",
    title:
      "Masquez le contenu sensible tout en préservant la structure du document",
    description:
      "Téléversez un PDF, un document Word ou une image. Le premier passage génère un fichier masqué provisoire pour révision. Ensuite, vous pouvez désélectionner tout groupe de texte que vous ne souhaitez pas masquer avant de générer le fichier final.",
    uploadTitle: "Téléverser un document à masquer",
    allowedFileInputs: "Autorisés : .pdf, .docx, .jpg, .jpeg, .png.",
    outputExtensionWillBe: "L’extension de sortie sera",
    unsupportedFileType:
      "Type de fichier non pris en charge : {ext}. Seuls .pdf, .docx, .jpg, .jpeg et .png sont autorisés.",
    fileTooLarge:
      "Le fichier est trop volumineux. La taille maximale autorisée est de {maxSize} MB.",
    chooseFileToMask: "Veuillez choisir un fichier à masquer.",
    maskingFailed:
      "Une erreur s’est produite pendant le traitement du masquage.",
    maskAction: "Masquer le document",
    generating: "Masquage...",
    reviewing: "Préparation de la révision...",
    finalizing: "Finalisation...",
    processAndReview: "Traiter et réviser",
    finalizeAction: "Générer le fichier final",
    resultTitle: "Sortie du masquage",
    previewEmpty:
      "Votre fichier masqué provisoire ou final apparaîtra ici avec les éléments groupés à réviser.",
    policySubtitle: "Règles strictes de traitement confidentiel",
    allowedUploadsLabel: "Téléversements autorisés :",
    outputRuleLabel: "Règle de sortie :",
    outputRuleValue:
      "l’extension de sortie doit toujours être identique à l’extension d’entrée",
    docTypeLabel: "Type de document",
    sensitiveTargetsLabel: "Données sensibles à masquer",
    exclusionsLabel: "Exclusions manuelles de révision",
    exclusionsPlaceholder:
      "Optionnel : saisissez les mots ou expressions à toujours laisser visibles, une par ligne ou séparés par des virgules.",
    selectedTargetsLabel: "Cibles sélectionnées :",
    fileAcceptedLabel: "Fichier accepté",
    downloadReady: "Téléchargement prêt",
    outputReadyText: "Votre fichier masqué final est prêt à être téléchargé.",
    missingDownloadUrl:
      "Le traitement est terminé, mais le backend n’a pas renvoyé d’URL de téléchargement.",
    processedFile: "Fichier traité",
    inputFile: "Fichier d’entrée",
    inputExtension: "Extension d’entrée",
    outputExtension: "Extension de sortie",
    documentTypeResult: "Type de document",
    exclusionsCount: "Nombre d’exclusions",
    rulesApplied:
      "Le masquage a été généré à partir de l’ensemble sélectionné après révision.",
    fileTypeLabel: "Type détecté :",
    selectAll: "Tout sélectionner",
    clearAll: "Tout effacer",

    provisionalReady: "Fichier masqué provisoire prêt",
    finalReady: "Fichier masqué final prêt",
    reviewTitle: "Réviser les éléments groupés à masquer",
    reviewHint:
      "Les éléments cochés restent masqués partout où ils apparaissent. Les éléments décochés restent visibles partout où ils apparaissent.",
    reviewItemsLabel: "Éléments groupés à réviser",
    processedPreviewTitle: "Aperçu du document traité",
    docxPreviewNotice:
      "L’aperçu DOCX est affiché à l’aide d’un aperçu PDF généré pour la révision. Le téléchargement final reste en DOCX.",
    approveAll: "Tout approuver",
    clearApproved: "Tout effacer",
    approvedCountLabel: "Éléments approuvés",
    deselectedCountLabel: "Éléments désélectionnés",
    occurrencesLabel: "Occurrences",
    noCandidates:
      "Aucun élément sensible groupé n’a été détecté pour les paramètres actuels.",
  },
};