<template>
<v-app :dark="dark">
  <v-content>
    <router-view @toggleTheme="dark = $event"></router-view>
  </v-content>
</v-app>
</template>

<script>
export default {
  name: 'App',
  data: () => ({
    dark: false
  })
}
</script>

<style>
  svg {
    user-select: none;
  }
</style>

