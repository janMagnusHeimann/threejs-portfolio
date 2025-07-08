export const navLinks = [{
        id: 1,
        name: 'Home',
        href: '#home',
    },
    {
        id: 2,
        name: 'About',
        href: '#about',
    },
    {
        id: 3,
        name: 'Work',
        href: '#work',
    },
    {
        id: 4,
        name: 'Contact',
        href: '#contact',
    },
];

export const publications = [{
        id: 1,
        title: 'Reaction Graph Networks for Inorganic Synthesis Condition Prediction of Solid State Materials',
        authors: 'Heimann, J., et al.',
        venue: 'AI4Mat-2024: NeurIPS 2024 Workshop on AI for Accelerated Materials Design',
        year: '2024',
        abstract: 'We present a novel approach using Graph Neural Networks with attention mechanisms to predict inorganic material synthesis conditions. Our method achieves significant improvements in accuracy over baseline approaches by leveraging the structural relationships in reaction graphs and incorporating domain-specific material science knowledge.',
        image: '/assets/publication1.png', // You can add publication images here
        pdf: 'https://openreview.net/forum?id=VGsXQOTs1E',
        tags: ['Graph Neural Networks', 'Materials Science', 'Deep Learning', 'Synthesis Prediction', 'AI4Science']
    }
    // Add more publications as they become available
];

export const myProjects = [{
        title: 'AutoApply - AI Job Application SaaS',
        desc: 'A revolutionary multi-agent system that automates job applications using GPT-4 and Claude-3 APIs. AutoApply has generated $480K ARR with 10K+ monthly active users by intelligently applying to relevant positions.',
        subdesc: 'Built with fine-tuned YOLOv8 for form detection achieving 94.3% accuracy, processing 78K+ successful applications. Scaled infrastructure handles 2.8M+ monthly queries with 99.7% uptime using containerized microservices.',
        href: 'https://github.com/janMagnusHeimann/autoapply-turbo-charge-jobs',
        texture: '/textures/project/project1.mp4',
        logo: '/assets/autoapply.png',
        logoStyle: {
            backgroundColor: '#2A1816',
            border: '0.2px solid #36201D',
            boxShadow: '0px 0px 60px 0px #AA3C304D',
        },
        spotlight: '/assets/spotlight1.png',
        tags: [{
                id: 1,
                name: 'GPT-4',
                path: '/assets/claude.png',
            },
            {
                id: 2,
                name: 'React',
                path: '/assets/react.svg',
            },
            {
                id: 3,
                name: 'Python',
                path: '/assets/python.png',
            },
            {
                id: 4,
                name: 'TypeScript',
                path: '/assets/typescript.png',
            },
        ],
    },
    {
        title: 'OpenRLHF Fork - Scalable RLHF Framework',
        desc: 'Enhanced OpenRLHF framework implementing hybrid DPO/PPO training pipeline with significant performance improvements. Reduced GPU memory usage by 15% and achieved 23% faster convergence on reward model training.',
        subdesc: 'Contributed multi-node distributed training support using DeepSpeed ZeRO-3, enabling training of 13B parameter models on 8x A100 clusters. Implemented adaptive KL penalty scheduling and batch-wise advantage normalization.',
        href: 'https://github.com/janMagnusHeimann/OpenRLHF',
        texture: '/textures/project/project2.mp4',
        logo: '/assets/project-logo2.png',
        logoStyle: {
            backgroundColor: '#13202F',
            border: '0.2px solid #17293E',
            boxShadow: '0px 0px 60px 0px #2F6DB54D',
        },
        spotlight: '/assets/spotlight2.png',
        tags: [{
                id: 1,
                name: 'PyTorch',
                path: '/assets/pytorch.png',
            },
            {
                id: 2,
                name: 'RLHF',
                path: '/assets/huggingface.png',
            },
            {
                id: 3,
                name: 'DeepSpeed',
                path: '/assets/python.png',
            },
            {
                id: 4,
                name: 'PPO/DPO',
                path: '/assets/claude.png',
            },
        ],
    },
    {
        title: 'ArchUnit TypeScript - Architecture Testing',
        desc: 'Open source TypeScript architecture testing library with 400+ GitHub stars and widespread adoption in the JavaScript ecosystem. Enables developers to validate architectural rules and maintain code quality at scale.',
        subdesc: 'Implemented AST-based static analysis supporting circular dependency detection, layered architecture validation, and code metrics. Built pattern matching system with glob/regex support and universal testing framework integration.',
        href: 'https://github.com/LukasNiessen/ArchUnitTS',
        texture: '/textures/project/project3.mp4',
        logo: '/assets/typescript.png',
        logoStyle: {
            backgroundColor: '#60f5a1',
            background: 'linear-gradient(0deg, #60F5A150, #60F5A150), linear-gradient(180deg, rgba(255, 255, 255, 0.9) 0%, rgba(208, 213, 221, 0.8) 100%)',
            border: '0.2px solid rgba(208, 213, 221, 1)',
            boxShadow: '0px 0px 60px 0px rgba(35, 131, 96, 0.3)',
        },
        spotlight: '/assets/spotlight3.png',
        tags: [{
                id: 1,
                name: 'TypeScript',
                path: '/assets/typescript.png',
            },
            {
                id: 2,
                name: 'AST Analysis',
                path: '/assets/terminal.png',
            },
            {
                id: 3,
                name: 'Jest',
                path: '/assets/react.svg',
            },
            {
                id: 4,
                name: 'Open Source',
                path: '/assets/github.svg',
            },
        ],
    },
    {
        title: 'Domain-Specific GPT-2 Fine-Tuning',
        desc: 'Fine-tuned GPT-2 medium on 10K aerospace papers using custom tokenizer with domain-specific vocabulary extensions. Achieved significant improvements in technical summarization capabilities.',
        subdesc: 'Implemented distributed training across 4 GPUs using gradient accumulation and achieved 12% ROUGE score improvement for technical summarization through careful hyperparameter tuning and data augmentation.',
        href: 'https://github.com/jan-heimann/gpt2-aerospace-finetuning',
        texture: '/textures/project/project4.mp4',
        logo: '/assets/project-logo4.png',
        logoStyle: {
            backgroundColor: '#0E1F38',
            border: '0.2px solid #0E2D58',
            boxShadow: '0px 0px 60px 0px #2F67B64D',
        },
        spotlight: '/assets/spotlight4.png',
        tags: [{
                id: 1,
                name: 'GPT-2',
                path: '/assets/claude.png',
            },
            {
                id: 2,
                name: 'HuggingFace',
                path: '/assets/huggingface.png',
            },
            {
                id: 3,
                name: 'PyTorch',
                path: '/assets/pytorch.png',
            },
            {
                id: 4,
                name: 'NLP',
                path: '/assets/python.png',
            },
        ],
    },
];

export const calculateSizes = (isSmall, isMobile, isTablet) => {
    return {
        deskScale: isSmall ? 0.05 : isMobile ? 0.06 : 0.065,
        deskPosition: isMobile ? [0.5, -4.5, 0] : [0.25, -5.5, 0],
        reactLogoPosition: isSmall ? [3, 4, 0] : isMobile ? [5, 4, 0] : isTablet ? [5, 4, 0] : [12, 3, 0],
        ringPosition: isSmall ? [-5, 7, 0] : isMobile ? [-10, 10, 0] : isTablet ? [-12, 10, 0] : [-24, 10, 0],
        // Tech logo positions
        pythonPosition: isSmall ? [2, 4, 0] : isMobile ? [4, 4, 0] : isTablet ? [4, 4, 0] : [10, 3, 0],
        huggingfacePosition: isSmall ? [4, -2, 0] : isMobile ? [6, -2, 0] : isTablet ? [7, -2, 0] : [10, -2, 0],
        mongodbPosition: isSmall ? [-4, 1, 0] : isMobile ? [-6, 1, 0] : isTablet ? [-8, 1, 0] : [-14, 1, 0],
    };
};

export const workExperiences = [{
        id: 1,
        name: 'DRWN AI',
        pos: 'Machine Learning Engineer',
        duration: 'Apr 2025 - Present',
        title: "Developing Multi-Agent Reinforcement Learning systems using PPO to optimize advertising budget allocation, achieving 15-25% improvement in cost-per-acquisition across client campaigns. Built real-time inference pipeline serving RL policies with 95ms latency.",
        icon: '/assets/framer.svg',
        animation: 'victory',
    },
    {
        id: 2,
        name: 'Rocket Factory Augsburg',
        pos: 'Machine Learning Engineer',
        duration: 'Mar 2024 - Mar 2025',
        title: "Designed RL pipeline using PPO to optimize rocket design parameters, achieving $1.5M projected cost reduction per launch. Implemented Graph Neural Networks to encode rocket component relationships and created custom OpenAI Gym environments.",
        icon: '/assets/rfa.png',
        animation: 'clapping',
    },
    {
        id: 3,
        name: 'MIT',
        pos: 'Assistant ML Researcher',
        duration: 'May 2024 - Dec 2024',
        title: "Developed Graph Neural Networks with attention mechanisms for material synthesis prediction, improving accuracy by 9.2% over baseline methods. Implemented multi-task transformer pretraining on 500K material descriptions.",
        icon: '/assets/mit.png',
        animation: 'salute',
    },
    {
        id: 4,
        name: 'Deepmask GmbH',
        pos: 'ML Engineer/Advisor',
        duration: 'Oct 2024 - Mar 2025',
        title: "Fine-tuned DeepSeek R1 (70B parameters) using LoRA with rank-16 adaptation, achieving +4% BLEU and +6% ROUGE-L on German benchmarks. Implemented production RAG system with 92% retrieval accuracy.",
        icon: '/assets/deepmask.png',
        animation: 'idle',
    },
    {
        id: 5,
        name: 'OHB Systems AG',
        pos: 'Software Engineer',
        duration: 'Jan 2023 - Mar 2024',
        title: "Built ML pipeline automating FEM analysis using Gaussian Processes, reducing engineering cycle time by 25%. Developed LSTM-based anomaly detection for satellite telemetry data and deployed models using MLflow and Docker.",
        icon: '/assets/ohb.png',
        animation: 'salute',
    },
    {
        id: 6,
        name: 'GetMoBie GmbH',
        pos: 'Co-Founder/Software Lead',
        duration: 'Jan 2021 - Dec 2022',
        title: "Led development of mobile banking application serving 20K+ users, presented at 'Die Höhle der Löwen' TV show. Implemented Random Forest models for transaction categorization and fraud detection on 1M+ records.",
        icon: '/assets/getmobie.png',
        animation: 'clapping',
    },
];