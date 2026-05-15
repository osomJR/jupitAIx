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
        description: "Convert speech to accurate text",
      },
      {
        key: "questions",
        name: "Generate Questions",
        route: "/questions",
        description: "Create smart questions from notes, text, or topics.",
      },
      {
        key: "redact",
        name: "Redaction",
        route: "/redact",
        description:
          "Redact sensitive data and information from documents preserving the original layout and structure",
      },
      {
        key: "mask",
        name: "Data Masking",
        route: "/mask",
        description:
          "Mask sensitive data and information keeping the document readable and usable",
      },
      {
        key: "compliance",
        name: "Compliance",
        route: "/compliance",
        description: "Check documents against compliance rules",
      },
      {
        key: "extraction",
        name: "Structured Extraction",
        route: "/extraction",
        description: "Extract key data from documents",
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
        description: "Convertir la parole en texte avec précision",
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
        name: "Caviardage",
        route: "/redact",
        description:
          "Caviardez les données et informations sensibles dans vos documents tout en préservant la mise en page et la structure d’origine",
      },
      {
        key: "mask",
        name: "Masquage des données",
        route: "/mask",
        description:
          "Masquez les données et informations sensibles tout en conservant un document lisible et exploitable",
      },
      {
        key: "compliance",
        name: "Conformité",
        route: "/compliance",
        description: "Vérifiez les documents selon les règles de conformité",
      },
      {
        key: "extraction",
        name: "Extraction structurée",
        route: "/extraction",
        description: "Extrayez les données clés des documents",
      },
    ],
  },
};
export const convertPageTranslations = {
  en: {
    badge: "Convert documents, files and images",
    title: "Convert files across several formats",
    description: "Upload PDF, Word document, JPG, JPEG, or PNG",
    uploadTitle: "Upload file or document",
    conversionOutput: "Conversion result",
    previewText: "Download appears here after file conversion",

    unsupportedFileType:
      "Unsupported file type: {ext}. Only .pdf, .docx, .jpg, .jpeg, and .png are allowed",
    fileTooLarge: "File is too large, maximum allowed size is {maxSize} MB",
    chooseFileToConvert: "Please choose a file to convert",
    invalidConversion: "This conversion combination is not allowed",
    conversionFailed: "Something went wrong while converting the file",
    missingDownloadUrl:
      "Conversion finished, but the backend did not return a download URL",

    conversionCompleted: "Conversion completed",
    inputFile: "Input file",
    inputExtension: "Input extension",
    outputExtension: "Output extension",
    downloadReady: "Download ready",
    convertedFile: "Converted file",
    outputReadyText: "Your converted file is ready for download",
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
      "Téléversez un PDF, un document Word, un JPG, un JPEG ou un PNG",
    uploadTitle: "Téléverser un fichier ou un document",
    conversionOutput: "Résultat de la conversion",
    previewText:
      "Le téléchargement apparaîtra ici après la conversion du fichier",

    unsupportedFileType:
      "Type de fichier non pris en charge: {ext}. Seuls les formats .pdf, .docx, .jpg, .jpeg et .png sont autorisés",
    fileTooLarge:
      "Le fichier est trop volumineux, la taille maximale autorisée est de {maxSize} Mo",
    chooseFileToConvert: "Veuillez choisir un fichier à convertir",
    invalidConversion: "Cette combinaison de conversion n’est pas autorisée",
    conversionFailed:
      "Une erreur s’est produite lors de la conversion du fichier",
    missingDownloadUrl:
      "La conversion est terminée, mais le backend n’a pas renvoyé d’URL de téléchargement",

    conversionCompleted: "Conversion terminée",
    inputFile: "Fichier d’entrée",
    inputExtension: "Extension d’entrée",
    outputExtension: "Extension de sortie",
    downloadReady: "Téléchargement prêt",
    convertedFile: "Fichier converti",
    outputReadyText: "Votre fichier converti est prêt à être téléchargé",
    detectedType: "Type détecté:",
    from: "De",
    convertTo: "Convertir vers",
    allowedOutputsFor: "Sorties autorisées pour",
    none: "aucune",
    conversionLabel: "Conversion:",

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
    description: "Upload audio (.mp3) or video (.mp4, .mkv, .mov)",
    uploadTitle: "Upload audio or video",
    allowedFileInputs: "Allowed inputs: .mp3, .mp4, .mkv, .mov",
    unsupportedFileType: "Unsupported file type: {ext}",
    fileTooLarge:
      "File is too large, maximum size for this media type is {maxSize} MB",
    mediaTooLong:
      "Media is too long, maximum duration for this media type is {maxDuration}",
    couldNotReadDuration:
      "Could not read media duration, Please try another file",
    chooseFileToTranscribe: "Please choose an audio or video file",
    transcriptionFailed: "Transcription request failed",
    validatingMedia: "Checking media",
    transcriptOutput: "Transcript output",
    previewText: "Your transcript will appear here after processing",

    transcriptOptionsTitle: "Transcription options",
    transcriptOptionsSubtitle: "Choose how the transcript should be processed",
    preserveFillerWordsLabel: "Preserve filler words",
    preserveFillerWordsHelp: "Keep words like “um”, “uh”, and similar fillers",
    removeBackgroundNoiseLabel: "Remove background noise",
    removeBackgroundNoiseHelp:
      "Apply optional minimal background-noise cleanup",
    diarizeSpeakersLabel: "Separate speakers",
    diarizeSpeakersHelp:
      "Separate speakers only when they are acoustically detectable",

    transcriptReady: "Transcript ready",
    transcriptReadyText:
      "The spoken content has been converted into written text",
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
      "Téléversez un fichier audio (.mp3) ou vidéo (.mp4, .mkv, .mov)",
    uploadTitle: "Téléverser un fichier audio ou vidéo",
    allowedFileInputs: "Entrées autorisées: .mp3, .mp4, .mkv, .mov",
    unsupportedFileType: "Type de fichier non pris en charge: {ext}",
    fileTooLarge:
      "Le fichier est trop volumineux, la taille maximale pour ce type de média est de {maxSize} Mo",
    mediaTooLong:
      "Le média est trop long, la durée maximale pour ce type de média est de {maxDuration}",
    couldNotReadDuration:
      "Impossible de lire la durée du média, veuillez essayer un autre fichier",
    chooseFileToTranscribe: "Veuillez choisir un fichier audio ou vidéo",
    transcriptionFailed: "La requête de transcription a échoué",
    validatingMedia: "Vérification du média",
    transcriptOutput: "Résultat de la transcription",
    previewText: "Votre transcription apparaîtra ici après le traitement",

    transcriptOptionsTitle: "Options de transcription",
    transcriptOptionsSubtitle:
      "Choisissez comment la transcription doit être traitée",
    preserveFillerWordsLabel: "Préserver les mots de remplissage",
    preserveFillerWordsHelp:
      "Conserver les mots comme « euh », « hum » et équivalents",
    removeBackgroundNoiseLabel: "Réduire le bruit de fond",
    removeBackgroundNoiseHelp:
      "Appliquer un nettoyage minimal et optionnel du bruit de fond",
    diarizeSpeakersLabel: "Séparer les intervenants",
    diarizeSpeakersHelp:
      "Séparer les intervenants uniquement lorsqu’ils sont détectables acoustiquement",

    transcriptReady: "Transcription prête",
    transcriptReadyText: "Le contenu parlé a été converti en texte écrit",
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
    title: "Redact sensitive data and information",
    description:
      "Upload files or documents to redact sensitive data and information keeping its structure",
    uploadTitle: "Upload file or document",
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
    exclusionsLabel: "Type characters to redact",
    exclusionsPlaceholder:
      "Optional: enter exact words, names, phrases, or characters to redact, one per line or comma-separated.",
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
    exclusionsCount: "Items left visible count",
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
    badge: "Caviardage en boîte noire axé sur la confidentialité",
    title: "Caviarder les données et informations sensibles",
    description:
      "Téléversez des fichiers ou des documents pour caviarder les données et informations sensibles tout en conservant leur structure",
    uploadTitle: "Téléverser un fichier ou un document",
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
    exclusionsLabel: "Saisir les caractères à caviarder",
    exclusionsPlaceholder:
      "Optionnel : saisissez les mots, noms, expressions ou caractères exacts à caviarder, une entrée par ligne ou séparée par des virgules.",
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
    exclusionsCount: "Nombre d’éléments laissés visibles",
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
    badge: "Privacy-first black-box data masking",
    title: "Mask sensitive data and information",
    description:
      "Upload files or documents to mask sensitive data and information keeping its structure",
    uploadTitle: "Upload file or document",
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
    exclusionsLabel: "Type characters to mask",
    exclusionsPlaceholder:
      "Optional: enter words, characters, or numbers to mask, one per line or comma-separated.",
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
    exclusionsCount: "Items left visible count",
    customMaskCount: "Custom mask items",
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
    badge: "Masquage en boîte noire axé sur la confidentialité",
    title: "Masquer les données et informations sensibles",
    description:
      "Téléversez des fichiers ou des documents pour masquer les données et informations sensibles tout en conservant leur structure",
    uploadTitle: "Téléverser un fichier ou un document",
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
    exclusionsLabel: "Saisir les caractères à masquer",
    exclusionsPlaceholder:
      "Optionnel : saisissez les mots, caractères ou nombres à masquer, une par ligne ou séparés par des virgules.",
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
    exclusionsCount: "Nombre d’éléments laissés visibles",
    customMaskCount: "Éléments personnalisés à masquer",
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
export const structuredExtractionPageTranslations = {
  en: {
    badge: "Structured document extraction",
    title: "Extract structured data from documents",
    description:
      "Upload files or document and export extracted fields, tables, and records",
    uploadTitle: "Upload document to extract",
    allowedFileInputs: "Allowed inputs: .pdf, .docx, .jpg, .jpeg, .png",
    extractionOutput: "Extraction result",
    previewText:
      "Your structured extraction file will appear here after processing",
    extractAction: "Extract data",
    extracting: "Extracting",
    extractionLabel: "Extraction:",
    extractionCompleted: "Structured extraction completed",

    unsupportedFileType:
      "Unsupported file type: {ext}. Only .pdf, .docx, .jpg, .jpeg, and .png are allowed",
    fileTooLarge: "File is too large, maximum allowed size is {maxSize} MB",
    chooseFileToExtract: "Please choose a file to extract from",
    documentClassRequired: "Select at least one document type",
    extractionFailed: "Something went wrong while extracting structured data",
    missingDownloadUrl:
      "Extraction finished, but the backend did not return a download URL",

    detectedType: "Detected type:",
    outputFormatLabel: "Download format",
    outputFormatHelp: "Choose the file format you want to download",
    outputFormatExamples:
      "Examples: JSON for apps, CSV for spreadsheets, Excel for review workbooks",
    resultShapeLabel: "Output layout",
    resultShapeHelp: "Choose how the extracted data should be organized",
    resultShapeExamples: "Not sure? Use Complete structured output",
    documentClassesLabel: "Document types",
    documentClassesHelp:
      "Select the document type that best matches your file, select more than one only if the file combines document types",
    documentClassesEmptyHelp: "At least one document type is required",
    documentClassesExamples:
      "Examples: Invoice, Bank statement, Contract, KYC document",
    searchDocumentClassesPlaceholder: "Search document types",
    selectedFieldsLabel: "Fields to extract",
    selectedFieldsHelp:
      "Optional, leave empty to extract all detected fields and add fields only when you know exactly what you need",
    selectedFieldsExamples: "Examples: invoice_number, invoice_date, total",
    selectedFieldsPlaceholder:
      "Optional, enter fields to extract and leave empty to extract detected fields",
    suggestedFieldsLabel: "Suggested fields",
    suggestedFieldsHelp:
      "Click common fields for the selected document type, suggestions are intentionally short to keep the page simple",
    suggestedFieldsExamples: "You can still type any custom field above",
    clearFields: "Clear fields",

    inputFile: "Input file",
    inputExtension: "Input extension",
    documentClassesResult: "Document types",
    resultShapeResult: "Output layout",
    outputFormatResult: "Download format",
    selectedFieldsResult: "Fields to extract",
    extractedFile: "Extracted file",
    allDetectedFields: "All detected fields",
    outputReadyText: "Your structured extraction file is ready to download",
    humanReviewRequired:
      "Human review is required before relying on or exporting the extracted data",
    downloadReady: "Download ready",

    outputFormatsTitle: "Outputs",
    reviewTitle: "Review",
    reviewValue: "Required",
    knowledgeTitle: "Knowledge",
    knowledgeValue: "Source-only",

    pdfDocument: "PDF document",
    wordDocument: "Word document",
    jpgImage: "JPG image",
    jpegImage: "JPEG image",
    pngImage: "PNG image",
    unknownFile: "Unknown file",

    outputFormatLabels: {
      json: ".json",
      csv: ".csv",
      xlsx: ".xlsx",
    },

    resultShapeLabels: {
      machine_readable: "Complete structured output (recommended)",
      key_value_fields: "Simple field/Value list",
      tables: "Extract tables only",
      row_based_records: "Spreadsheet/Database rows",
    },

    resultShapeDescriptions: {
      machine_readable:
        "Default includes fields, tables, records, warnings, and evidence in a complete structured file",
      key_value_fields:
        "Best for forms, invoices, receipts, IDs, HR records, and other documents with named fields",
      tables:
        "Best when the document contains visible tables and you mainly want table rows",
      row_based_records:
        "Best for repeated items such as transactions, line items, clauses, tickets, or database-ready rows",
    },

    documentClassLabels: {
      form: "Form",
      memo: "Memo",
      invoice: "Invoice",
      receipt: "Receipt",
      bank_statement: "Bank statement",
      kyc_document: "KYC document",
      id_document: "ID document",
      contract: "Contract",
      legal_record: "Legal record",
      medical_record: "Medical record",
      procurement_document: "Procurement document",
      technical_report: "Technical report",
      incident_report: "Incident report",
      insurance_document: "Insurance document",
      hr_record: "HR record",
      onboarding_document: "Onboarding document",
      ticket: "Ticket",
    },
  },

  fr: {
    badge: "Extraction structurée de documents",
    title: "Extraire des données structurées de documents",
    description:
      "Téléversez des fichiers ou documents et exportez les champs, tableaux et enregistrements extraits",
    uploadTitle: "Téléverser un document à extraire",
    allowedFileInputs: "Entrées autorisées: .pdf, .docx, .jpg, .jpeg, .png",
    extractionOutput: "Résultat de l’extraction",
    previewText:
      "Votre fichier d’extraction structurée apparaîtra ici après le traitement",
    extractAction: "Extraire les données",
    extracting: "Extraction",
    extractionLabel: "Extraction:",
    extractionCompleted: "Extraction structurée terminée",

    unsupportedFileType:
      "Type de fichier non pris en charge: {ext}. Seuls les formats .pdf, .docx, .jpg, .jpeg et .png sont autorisés",
    fileTooLarge:
      "Le fichier est trop volumineux, la taille maximale autorisée est de {maxSize} Mo",
    chooseFileToExtract: "Veuillez choisir un fichier à extraire",
    documentClassRequired: "Sélectionnez au moins un type de document",
    extractionFailed:
      "Une erreur s’est produite lors de l’extraction des données structurées",
    missingDownloadUrl:
      "L’extraction est terminée, mais le backend n’a pas renvoyé d’URL de téléchargement",

    detectedType: "Type détecté:",
    outputFormatLabel: "Format du téléchargement",
    outputFormatHelp: "Choisissez le format du fichier à télécharger",
    outputFormatExamples:
      "Exemples : JSON pour les applications, CSV pour les feuilles de calcul, Excel pour les classeurs de révision",
    resultShapeLabel: "Organisation de la sortie",
    resultShapeHelp:
      "Choisissez comment les données extraites doivent être organisées",
    resultShapeExamples:
      "Vous hésitez ? Utilisez la sortie structurée complète",
    documentClassesLabel: "Types de document",
    documentClassesHelp:
      "Sélectionnez le type qui correspond le mieux au fichier, sélectionnez plusieurs types seulement si le fichier combine réellement plusieurs documents",
    documentClassesEmptyHelp: "Au moins un type de document est requis",
    documentClassesExamples:
      "Exemples: Facture, Relevé bancaire, Contrat, Document KYC",
    searchDocumentClassesPlaceholder: "Rechercher des types de document",
    selectedFieldsLabel: "Champs à extraire",
    selectedFieldsHelp:
      "Facultatif, laisser vide pour extraire tous les champs détectés et n'ajouter des champs que lorsque vous savez exactement ce dont vous avez besoin",
    selectedFieldsExamples: "Exemples: invoice_number, invoice_date, total",
    selectedFieldsPlaceholder:
      "Facultatif, saisissez les champs à extraire ou laissez vide pour extraire les champs détectés",
    suggestedFieldsLabel: "Champs suggérés",
    suggestedFieldsHelp:
      "Cliquez sur les champs courants pour le type de document sélectionné, les suggestions sont volontairement courtes pour garder la page simple",
    suggestedFieldsExamples:
      "Vous pouvez toujours saisir un champ personnalisé ci-dessus",
    clearFields: "Effacer les champs",

    inputFile: "Fichier d’entrée",
    inputExtension: "Extension d’entrée",
    documentClassesResult: "Types de document",
    resultShapeResult: "Organisation de la sortie",
    outputFormatResult: "Format du téléchargement",
    selectedFieldsResult: "Champs à extraire",
    extractedFile: "Fichier extrait",
    allDetectedFields: "Tous les champs détectés",
    outputReadyText:
      "Votre fichier d’extraction structurée est prêt à être téléchargé",
    humanReviewRequired:
      "Une révision humaine est requise avant de se fier aux données extraites ou de les exporter",
    downloadReady: "Téléchargement prêt",

    outputFormatsTitle: "Sorties",
    reviewTitle: "Révision",
    reviewValue: "Requise",
    knowledgeTitle: "Connaissance",
    knowledgeValue: "Source uniquement",

    pdfDocument: "Document PDF",
    wordDocument: "Document Word",
    jpgImage: "Image JPG",
    jpegImage: "Image JPEG",
    pngImage: "Image PNG",
    unknownFile: "Fichier inconnu",

    outputFormatLabels: {
      json: ".json",
      csv: ".csv",
      xlsx: ".xlsx",
    },

    resultShapeLabels: {
      machine_readable: "Sortie structurée complète (recommandé)",
      key_value_fields: "Liste simple champ/Valeur",
      tables: "Extraire uniquement les tableaux",
      row_based_records: "Lignes pour tableur/Base de données",
    },

    resultShapeDescriptions: {
      machine_readable:
        "Par défaut, les fichiers structurés complets comprennent les champs, les tableaux, les enregistrements, les avertissements et les preuves",
      key_value_fields:
        "Idéal pour les formulaires, factures, reçus, pièces d’identité, dossiers RH et documents avec champs nommés",
      tables:
        "Idéal lorsque le document contient des tableaux visibles et que vous voulez surtout les lignes du tableau",
      row_based_records:
        "Idéal pour les éléments répétés comme transactions, lignes de facture, clauses, tickets ou lignes prêtes pour une base de données",
    },

    documentClassLabels: {
      form: "Formulaire",
      memo: "Note",
      invoice: "Facture",
      receipt: "Reçu",
      bank_statement: "Relevé bancaire",
      kyc_document: "Document KYC",
      id_document: "Pièce d’identité",
      contract: "Contrat",
      legal_record: "Dossier juridique",
      medical_record: "Dossier médical",
      procurement_document: "Document d’approvisionnement",
      technical_report: "Rapport technique",
      incident_report: "Rapport d’incident",
      insurance_document: "Document d’assurance",
      hr_record: "Dossier RH",
      onboarding_document: "Document d’intégration",
      ticket: "Ticket",
    },
  },
};
export const compliancePageTranslations = {
  en: {
    badge: "Multi-country compliance checks",
    title: "Check documents against compliance rules",
    description:
      "Upload files or documents and generate a compliance report from configured jurisdiction and sector rule packs",
    uploadTitle: "Upload document to check",
    allowedFileInputs: "Allowed inputs: .pdf, .docx, .jpg, .jpeg, .png",
    complianceOutput: "Compliance result",
    previewText:
      "Your compliance report will appear here after the document is checked",
    checkAction: "Check compliance",
    checking: "Checking",
    complianceLabel: "Compliance:",
    complianceCompleted: "Compliance check completed",

    unsupportedFileType:
      "Unsupported file type: {ext}. Only .pdf, .docx, .jpg, .jpeg, and .png are allowed",
    fileTooLarge: "File is too large, maximum allowed size is {maxSize} MB",
    chooseFileToCheck: "Please choose a file to check",
    complianceFailed: "Something went wrong while checking compliance",
    corePackRequired:
      "The core control library for {country} must be included with every sector-specific compliance check",
    missingDownloadUrl:
      "Compliance check finished but the backend did not return a download URL",

    detectedType: "Detected type:",
    jurisdictionLabel: "Country/Jurisdiction",
    jurisdictionHelp:
      "Choose the country whose compliance rules should be used",
    jurisdictionExamples:
      "Examples: Nigeria, South Africa, United States, United Kingdom",
    reportVariantLabel: "Report format",
    reportVariantHelp: "Choose how the compliance result should be delivered",
    reportVariantExamples: "Not sure? Use PDF report for review",
    sectorPacksLabel: "Business sector/Rule packs",
    corePackHelp:
      "The core control library for {country} is always included, add sector-specific packs when needed",
    sectorPacksEmptyHelp:
      "Choose the sector that best matches the document or business context",
    sectorPacksExamples:
      "Examples: Banking and fintech, Health, Insurance, Telecom",
    searchSectorPacksPlaceholder: "Search sectors",
    clearSectorPacks: "Clear sectors",
    requiredLabel: "required",
    regulatoryDomainsLabel: "Focus areas",
    regulatoryDomainsHelp:
      "Optional, select focus areas only when you want a narrower review",
    regulatoryDomainsEmptyHelp:
      "Nothing selected means check all available domains in the selected rule packs",
    regulatoryDomainsExamples:
      "Examples: Privacy for personal data, AML for KYC/Fintech, Licensing for regulated businesses",
    searchRegulatoryDomainsPlaceholder: "Search focus areas",
    clearDomains: "Clear focus areas",

    inputFile: "Input file",
    inputExtension: "Input extension",
    jurisdictionResult: "Country/jurisdiction",
    sectorPacksResult: "Sector packs",
    regulatoryDomainsResult: "Focus areas",
    reportVariantResult: "Report format",
    outputFormatResult: "Output format",
    reportFile: "Report file",
    allDomains: "All available focus areas",
    outputReadyText: "Your compliance report is ready to download",
    humanReviewRequired:
      "Human review is required before relying on or exporting the compliance result",
    downloadReady: "Download ready",

    findingsSummary: "Findings summary",
    passed: "Passed",
    failed: "Failed",
    warning: "Warning",
    missing: "Missing",
    reviewRequiredCount: "Review required",
    reviewRequiredShort: "Review",

    outputTitle: "Output",
    reviewTitle: "Review",
    reviewValue: "Required",
    scopeTitle: "Scope",
    scopeValue: "Expandable by jurisdiction",

    pdfDocument: "PDF document",
    wordDocument: "Word document",
    jpgImage: "JPG image",
    jpegImage: "JPEG image",
    pngImage: "PNG image",
    unknownFile: "Unknown file",

    countryLabels: {
      nigeria: "Nigeria",
      unitedStates: "United States",
      unitedKingdom: "United Kingdom",
      southAfrica: "South Africa",
      canada: "Canada",
      france: "France",
      togo: "Togo",
      ghana: "Ghana",
    },

    outputFormatLabels: {
      pdf: ".pdf",
      json: ".json",
    },

    reportVariantLabels: {
      human_readable_report: "PDF report for review (recommended)",
      machine_readable_report: "JSON report for systems/API",
      annotated_source_output: "Evidence-marked PDF",
    },

    reportVariantDescriptions: {
      human_readable_report:
        "Best for reading, sharing, and downloading a normal compliance report",
      machine_readable_report:
        "Best for developers, dashboards, databases, APIs, or automated workflows",
      annotated_source_output:
        "Best when a reviewer needs to verify findings against the original source document",
    },

    sectorPackLabels: {
      nigeria_core_control_library: "Core control library",
      core_control_library: "Core control library",

      accounting: "Accounting",
      agriculture: "Agriculture",
      aviation: "Aviation",
      banking_and_fintech: "Banking and fintech",
      payment_platforms_and_services: "Payment platforms and services",
      energy_and_power: "Energy and power",
      health: "Health",
      insurance: "Insurance",
      legal_and_law: "Legal and law",
      law_and_legal: "Law and legal",
      manufacturing: "Manufacturing",
      maritime: "Maritime",
      maritime_and_shipping: "Maritime and shipping",
      media: "Media",
      mining: "Mining",
      ngo: "NGO",
      oil_and_gas: "Oil and gas",
      pharmaceuticals: "Pharmaceuticals",
      sports: "Sports",
      tech: "Technology",
      telecom: "Telecom",
    },

    regulatoryDomainLabels: {
      privacy: "Privacy/Personal data",
      cybersecurity: "Cybersecurity",
      aml: "AML/Financial crime",
      consumer_protection: "Consumer protection",
      public_sector_access_to_information:
        "Public-sector access to information",
      licensing: "Licensing/Permits",
      registration: "Registration",
      sector_regulator_requirements: "Sector regulator rules",
    },
  },

  fr: {
    badge: "Contrôles de conformité multi-pays",
    title: "Vérifier les documents selon les règles de conformité",
    description:
      "Téléversez des fichiers ou documents et générez un rapport de conformité à partir des packs de règles configurés par juridiction et par secteur",
    uploadTitle: "Téléverser un document à vérifier",
    allowedFileInputs: "Entrées autorisées: .pdf, .docx, .jpg, .jpeg, .png",
    complianceOutput: "Résultat de conformité",
    previewText:
      "Votre rapport de conformité apparaîtra ici après la vérification du document",
    checkAction: "Vérifier la conformité",
    checking: "Vérification",
    complianceLabel: "Conformité:",
    complianceCompleted: "Vérification de conformité terminée",

    unsupportedFileType:
      "Type de fichier non pris en charge: {ext}. Seuls les formats .pdf, .docx, .jpg, .jpeg et .png sont autorisés",
    fileTooLarge:
      "Le fichier est trop volumineux, la taille maximale autorisée est de {maxSize} Mo",
    chooseFileToCheck: "Veuillez choisir un fichier à vérifier",
    complianceFailed:
      "Une erreur s’est produite lors de la vérification de conformité",
    corePackRequired:
      "La bibliothèque de contrôles de base pour {country} doit être incluse avec chaque vérification sectorielle",
    missingDownloadUrl:
      "La vérification de conformité est terminée, mais le backend n’a pas renvoyé d’URL de téléchargement",

    detectedType: "Type détecté:",
    jurisdictionLabel: "Pays/Juridiction",
    jurisdictionHelp:
      "Choisissez le pays dont les règles de conformité doivent être utilisées",
    jurisdictionExamples:
      "Exemples: Nigeria, Afrique du Sud, États-Unis, Royaume-Uni",
    reportVariantLabel: "Format du rapport",
    reportVariantHelp:
      "Choisissez comment le résultat de conformité doit être livré",
    reportVariantExamples:
      "Vous hésitez? Utilisez le rapport PDF pour révision",
    sectorPacksLabel: "Secteur d’activité/Packs de règles",
    corePackHelp:
      "La bibliothèque de contrôles de base pour {country} est toujours incluse, ajoutez des packs sectoriels si nécessaire",
    sectorPacksEmptyHelp:
      "Choisissez le secteur qui correspond le mieux au document ou au contexte de l’entreprise",
    sectorPacksExamples:
      "Exemples : Banque et fintech, Santé, Assurance, Télécoms",
    searchSectorPacksPlaceholder: "Rechercher des secteurs",
    clearSectorPacks: "Effacer les secteurs",
    requiredLabel: "requis",
    regulatoryDomainsLabel: "Domaines de vérification",
    regulatoryDomainsHelp:
      "Optionnel, sélectionnez des domaines seulement si vous voulez une vérification plus ciblée",
    regulatoryDomainsEmptyHelp:
      "Rien n'est sélectionné signifie cocher tous les domaines disponibles dans les ensembles de règles sélectionnés",
    regulatoryDomainsExamples:
      "Exemples: Confidentialité pour les données personnelles, LBC pour KYC/Fintech, Licences pour les activités réglementées",
    searchRegulatoryDomainsPlaceholder: "Rechercher des domaines",
    clearDomains: "Effacer les domaines",

    inputFile: "Fichier d’entrée",
    inputExtension: "Extension d’entrée",
    jurisdictionResult: "Pays/Juridiction",
    sectorPacksResult: "Packs sectoriels",
    regulatoryDomainsResult: "Domaines de vérification",
    reportVariantResult: "Format du rapport",
    outputFormatResult: "Format de sortie",
    reportFile: "Fichier du rapport",
    allDomains: "Tous les domaines disponibles",
    outputReadyText: "Votre rapport de conformité est prêt à être téléchargé",
    humanReviewRequired:
      "Une révision humaine est requise avant de se fier au résultat de conformité ou de l’exporter",
    downloadReady: "Téléchargement prêt",

    findingsSummary: "Résumé des constats",
    passed: "Réussi",
    failed: "Échoué",
    warning: "Avertissement",
    missing: "Manquant",
    reviewRequiredCount: "Révision requise",
    reviewRequiredShort: "Révision",

    outputTitle: "Sortie",
    reviewTitle: "Révision",
    reviewValue: "Requise",
    scopeTitle: "Portée",
    scopeValue: "Extensible par juridiction",

    pdfDocument: "Document PDF",
    wordDocument: "Document Word",
    jpgImage: "Image JPG",
    jpegImage: "Image JPEG",
    pngImage: "Image PNG",
    unknownFile: "Fichier inconnu",

    countryLabels: {
      nigeria: "Nigeria",
      unitedStates: "États-Unis",
      unitedKingdom: "Royaume-Uni",
      southAfrica: "Afrique du Sud",
      canada: "Canada",
      france: "France",
      togo: "Togo",
      ghana: "Ghana",
    },

    outputFormatLabels: {
      pdf: ".pdf",
      json: ".json",
    },

    reportVariantLabels: {
      human_readable_report: "Rapport PDF pour révision (recommandé)",
      machine_readable_report: "Rapport JSON pour systèmes/API",
      annotated_source_output: "PDF avec preuves marquées",
    },

    reportVariantDescriptions: {
      human_readable_report:
        "Idéal pour lire, partager et télécharger un rapport de conformité normal",
      machine_readable_report:
        "Idéal pour les développeurs, tableaux de bord, bases de données, API ou workflows automatisés",
      annotated_source_output:
        "Idéal lorsqu’un réviseur doit vérifier les constats par rapport au document source original",
    },

    sectorPackLabels: {
      nigeria_core_control_library: "Bibliothèque de contrôles de base",
      core_control_library: "Bibliothèque de contrôles de base",

      accounting: "Comptabilité",
      agriculture: "Agriculture",
      aviation: "Aviation",
      banking_and_fintech: "Banque et fintech",
      payment_platforms_and_services: "Plateformes et services de paiement",
      energy_and_power: "Énergie et électricité",
      health: "Santé",
      insurance: "Assurance",
      legal_and_law: "Juridique et droit",
      law_and_legal: "Droit et juridique",
      manufacturing: "Fabrication",
      maritime: "Maritime",
      maritime_and_shipping: "Maritime et transport maritime",
      media: "Médias",
      mining: "Mines",
      ngo: "ONG",
      oil_and_gas: "Pétrole et gaz",
      pharmaceuticals: "Produits pharmaceutiques",
      sports: "Sports",
      tech: "Technologie",
      telecom: "Télécoms",
    },

    regulatoryDomainLabels: {
      privacy: "Confidentialité/Données personnelles",
      cybersecurity: "Cybersécurité",
      aml: "LBC/Criminalité financière",
      consumer_protection: "Protection des consommateurs",
      public_sector_access_to_information:
        "Accès à l’information du secteur public",
      licensing: "Licences/Permis",
      registration: "Enregistrement",
      sector_regulator_requirements: "Règles des régulateurs sectoriels",
    },
  },
};
