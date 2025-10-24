module.exports = {
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 2022,
    sourceType: 'module',
  },
  plugins: ['@typescript-eslint'],
  extends: [
    'eslint:recommended',
  ],
  env: {
    node: true,
    es2022: true,
  },
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
    // Allow console.log in server code
    'no-console': 'off',
    // Basic TypeScript rules
    '@typescript-eslint/no-explicit-any': 'warn',
    '@typescript-eslint/no-empty-function': 'warn',
    // Disable no-undef as TypeScript handles this
    'no-undef': 'off',
  },
  ignorePatterns: ['dist/', 'node_modules/', '*.js'],
};
