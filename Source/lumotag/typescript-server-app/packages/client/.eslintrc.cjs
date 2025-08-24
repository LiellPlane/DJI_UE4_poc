module.exports = {
  root: true,
  env: {
    browser: true,
    es2020: true,
  },
  extends: [
    'eslint:recommended',
  ],
  ignorePatterns: ['dist', '.eslintrc.cjs'],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    ecmaFeatures: {
      jsx: true,
    },
  },
  plugins: [
    '@typescript-eslint',
  ],
  rules: {
    // Allow unused parameters if they start with underscore
    'no-unused-vars': 'off',
    '@typescript-eslint/no-unused-vars': [
      'warn',
      {
        argsIgnorePattern: '^_',
        varsIgnorePattern: '^_',
      },
    ],
    // Allow any type in some cases (can be made stricter later)
    '@typescript-eslint/no-explicit-any': 'warn',
    // Allow empty functions
    '@typescript-eslint/no-empty-function': 'warn',
    // Allow console.log for debugging (can be changed to 'error' for production)
    'no-console': process.env.NODE_ENV === 'production' ? 'error' : 'off',
    // Allow unused imports for now
    'no-undef': 'off',
  },
  settings: {
    react: {
      version: '18.2',
    },
  },
};